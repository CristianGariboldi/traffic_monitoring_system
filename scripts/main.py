#!/usr/bin/env python3
"""
main_2.py - TWO-GATE (full-width) gate-based counting with AUTO-CALIBRATION + speed estimation

Keep all original behaviour; adds speed estimation via homography (preferred) or fallback car-length scale.
"""
import argparse
import time
import cv2
import numpy as np
from detector_onnx import ONNXDetector as Detector
from tracker import CentroidTracker
from speed_estimator import SpeedEstimator, compute_homography_from_pairs
import json, os

try:
    from filters import load_filter_from_config
    _HAS_FILTERS = True
except Exception:
    load_filter_from_config = None
    _HAS_FILTERS = False

try:
    from color_filter import is_white_car as legacy_is_white_car
except Exception:
    legacy_is_white_car = None

ALLOWED_CLASSES = {'car', 'truck', 'bus', 'van', 'motorbike', 'motorcycle'}


# # ###############3 USE THIS ONLY WITH BEST_YOLO.ONNX ######################3
# ALLOWED_CLASSES = {'bus', 'car', 'truck'}

# # 2. ADD THIS dictionary to define your custom model's class names
# # This maps the model's output IDs (0, 1, 2) to their names.
# CUSTOM_CLASS_NAMES = {
#     0: 'bus', 
#     1: 'car', 
#     2: 'truck'
# }

#################################################################

def parse_args():
    p = argparse.ArgumentParser(description="Two-gate counting with auto-calibration and config-driven filters")
    p.add_argument('--source', '-s', default="./data/Video.mp4", help='video file or camera index')
    p.add_argument('--model', default='./models/yolo11n.onnx', help='ONNX model path')
    p.add_argument('--conf', type=float, default=0.35, help='detection confidence threshold (0..1)')
    p.add_argument('--debug', action='store_true', help='print debug info')
    p.add_argument('--filter-config', default="./config/filters.yaml", help='path to filters YAML/JSON config')
    p.add_argument('--filter-name', default=None, help='named filter in config (if omitted, no filter => count all)')
    p.add_argument('--white-only', action='store_true', help='legacy: if set, count only white cars using old heuristic')
    p.add_argument('--auto-calib', action='store_true', help='auto-calibrate gate positions from first frames')
    p.add_argument('--calib-frames', type=int, default=60, help='frames for calibration if --auto-calib')
    p.add_argument('--calib-top-q', type=float, default=0.65, help='top quantile for gate placement (0..1)')
    p.add_argument('--calib-bottom-q', type=float, default=0.80, help='bottom quantile for gate placement (0..1)')
    p.add_argument('--seq-window', type=float, default=3.0, help='seconds allowed between two gate entries')
    p.add_argument('--fps-sync', action='store_true', help='match display to source FPS')
    p.add_argument('--homography', default="./config/homography.json",
                help='path to homography JSON file with image_points and world_points (meters)')
    p.add_argument('--speed-smooth-alpha', type=float, default=0.6,
                help='EMA alpha for speed smoothing (0..1). If 0 use median smoothing.')
    return p.parse_args()

def draw_box(frame, bbox, label=None, color=(0,255,0), thickness=2):
    x1,y1,x2,y2 = map(int, bbox)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, max(0,y1-18)), (x1+tw+6, y1), color, -1)
        cv2.putText(frame, label, (x1+2, max(0,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

def point_in_rect(pt, rect):
    x,y = pt
    x1,y1,x2,y2 = rect
    return (x >= x1 and x <= x2 and y >= y1 and y <= y2)

def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Cannot open source:", args.source)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_time = 1.0 / fps

    # create detector (ONNXDetector)
    ######################### USE THIS ONLY WITH BEST_YOLO.ONNX ##########################
    # detector = Detector(args.model, input_size=640, providers=['CPUExecutionProvider'], conf_thres=args.conf, class_names=CUSTOM_CLASS_NAMES, debug=args.debug)
    #########################################################

    detector = Detector(args.model, input_size=640, providers=['CPUExecutionProvider'], conf_thres=args.conf, debug=args.debug)

    tracker = CentroidTracker(max_missed=12, max_distance=140)

    filter_obj = None
    if args.filter_config and args.filter_name:
        if not _HAS_FILTERS:
            print("Warning: filters module not available. Ignoring --filter-config.")
        else:
            try:
                filter_obj = load_filter_from_config(args.filter_config, args.filter_name)
                print(f"Loaded filter '{args.filter_name}' from {args.filter_config}")
            except Exception as e:
                print("Failed to load filter config:", e)
                filter_obj = None

    legacy_white_mode = bool(args.white_only) and (filter_obj is None)

    count_in = 0
    count_out = 0
    seq_window = float(args.seq_window)

    frame_idx = 0
    ys = []   
    auto_calib = bool(args.auto_calib)
    calib_frames = max(1, int(args.calib_frames))
    top_q = float(args.calib_top_q)
    bottom_q = float(args.calib_bottom_q)

    default_top_frac = 0.65
    default_bottom_frac = 0.70

    top_y = None
    bottom_y = None

    prev_time = time.time()

    ########################################
    H_homography = None
    if args.homography:
        try:
            with open(args.homography, 'r') as fh:
                j = json.load(fh)
            img_pts = j['image_points']
            world_pts = j['world_points']
            H_homography = compute_homography_from_pairs(img_pts, world_pts)
            print("Loaded homography from", args.homography)
        except Exception as e:
            print("Failed to load homography:", e)
            H_homography = None

    fallback_m_per_px = None
    if H_homography is None and (args.ref_car_length is None or args.ref_car_length <= 0):
        fallback_m_per_px = None

    speed_est = SpeedEstimator(H=H_homography, fallback_m_per_px=fallback_m_per_px, history_len=8,
                            smooth_alpha=(args.speed_smooth_alpha if args.speed_smooth_alpha>0 else None))
    ######################################################

    while True:
        start_frame = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        h_img, w_img = frame.shape[:2]

    
        if auto_calib and frame_idx < calib_frames:
            dets_tmp = detector.detect(frame)
            for d in dets_tmp:
                cls = str(d['class_name']).lower()
                if cls not in ALLOWED_CLASSES:
                    continue
                x1,y1,x2,y2 = d['bbox']
                cy = int(y2)  
                ys.append(cy)
            frame_idx += 1

            cv2.putText(frame, f'Auto-calibrating gates: {frame_idx}/{calib_frames}', (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.line(frame, (0, int(h_img*default_top_frac)), (w_img-1, int(h_img*default_top_frac)), (200,200,0), 1)
            cv2.line(frame, (0, int(h_img*default_bottom_frac)), (w_img-1, int(h_img*default_bottom_frac)), (200,200,0), 1)

            cv2.imshow('output', frame)
            if args.fps_sync:
                elapsed = time.time() - start_frame
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            continue  

        if auto_calib and top_y is None and frame_idx >= calib_frames:
            if len(ys) >= 4:
                top_y = int(np.quantile(ys, top_q))
                bottom_y = int(np.quantile(ys, bottom_q))
                if top_y >= bottom_y:
                    med = int(np.median(ys))
                    span = max(10, int(0.12 * h_img))
                    top_y = max(8, med - span)
                    bottom_y = min(h_img-8, med + span)
                    if args.debug:
                        print("Calib quantiles were inverted; used median fallback.", med, span)
            else:
                top_y = int(h_img * default_top_frac)
                bottom_y = int(h_img * default_bottom_frac)
            auto_calib = False
            print(f"Auto-calibrated gates: top_y={top_y}, bottom_y={bottom_y}  (collected {len(ys)} samples)")

        if top_y is None:
            top_y = int(h_img * default_top_frac)
        if bottom_y is None:
            bottom_y = int(h_img * default_bottom_frac)

        gate_half_h = max(8, int(h_img * 0.02))
        top_gate = (0, max(0, top_y - gate_half_h), w_img - 1, min(h_img - 1, top_y + gate_half_h))
        bottom_gate = (0, max(0, bottom_y - gate_half_h), w_img - 1, min(h_img - 1, bottom_y + gate_half_h))

        
        dets = detector.detect(frame)
        all_dets_for_tracker = []
        for d in dets:
            cls = str(d['class_name']).lower()
            if cls not in ALLOWED_CLASSES:
                continue

            is_countable = True
            if filter_obj is not None:
                try:
                    is_countable = bool(filter_obj.match(frame, d['bbox'], cls))
                except Exception:
                    is_countable = False
            elif legacy_white_mode:
                if legacy_is_white_car is not None and cls == 'car':
                    is_countable = bool(legacy_is_white_car(frame, d['bbox']))
                else:
                    is_countable = False
            else:
                is_countable = True

            all_dets_for_tracker.append({'bbox': d['bbox'], 'class_name': cls, 'is_countable': is_countable})
            if args.debug and not is_countable:
                print(f"DEBUG skip countable: cls={cls}, bbox={d['bbox']}, is_countable={is_countable}")

        tracks = tracker.update(all_dets_for_tracker)

        cv2.rectangle(frame, (top_gate[0], top_gate[1]), (top_gate[2], top_gate[3]), (0,255,0), 2)    # top - green
        cv2.rectangle(frame, (bottom_gate[0], bottom_gate[1]), (bottom_gate[2], bottom_gate[3]), (0,0,255), 2)  # bottom - red

        video_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        ts = (video_msec / 1000.0) if video_msec and video_msec > 0 else time.time()

        if speed_est.H is None and speed_est.fallback_m_per_px is None and args.ref_car_length and args.ref_car_length > 0:
            for det in all_dets_for_tracker:
                if det['class_name'] == 'car':
                    x1,y1,x2,y2 = det['bbox']
                    pix_w = max(1, int(x2) - int(x1))
                    speed_est.fallback_m_per_px = float(args.ref_car_length) / float(pix_w)
                    if args.debug:
                        print("Set fallback m_per_px =", speed_est.fallback_m_per_px, "from pix width", pix_w)
                    break

        for t in tracks:
            if t.class_name == 'car':
                color = (0, 255, 0) if t.is_countable else (0, 255, 255)
            elif t.class_name in ('truck','bus','van'):
                color = (0, 165, 255)
            elif t.class_name in ('motorbike','motorcycle'):
                color = (255, 0, 255)
            else:
                color = (200,200,200)

            if args.debug:
                print(f"TRACK {t.id}: centroid={t.centroid} hits={t.hits} miss={t.miss} is_countable={t.is_countable} counted={t.counted}")

            inside_top_now = point_in_rect(t.centroid, top_gate)
            was_inside_top = ('top' in t.inside_gates)
            if inside_top_now and not was_inside_top:
                t.enter_gate('top', ts)
                if args.debug:
                    print(f"  Track {t.id} ENTER TOP at {ts:.2f}")
            elif not inside_top_now and was_inside_top:
                t.exit_gate('top')
                if args.debug:
                    print(f"  Track {t.id} EXIT TOP at {ts:.2f}")

            inside_bottom_now = point_in_rect(t.centroid, bottom_gate)
            was_inside_bottom = ('bottom' in t.inside_gates)
            if inside_bottom_now and not was_inside_bottom:
                t.enter_gate('bottom', ts)
                if args.debug:
                    print(f"  Track {t.id} ENTER BOTTOM at {ts:.2f}")
            elif not inside_bottom_now and was_inside_bottom:
                t.exit_gate('bottom')
                if args.debug:
                    print(f"  Track {t.id} EXIT BOTTOM at {ts:.2f}")

            x1,y1,x2,y2 = t.bbox
            img_pt = ( (float(x1) + float(x2)) / 2.0, float(y2) )
            speed_kmh = speed_est.add_observation(t.id, img_pt, ts=ts)
            if speed_kmh is not None:
                t.speed_kmh = speed_kmh
            else:
                t.speed_kmh = None

            if not t.counted:
                recent = [ (g, tt) for g, tt in t.gate_history if (ts - tt) <= seq_window ]
                if args.debug and recent:
                    print(f"  Track {t.id} recent gates: {recent}")

                if len(recent) >= 2:
                    prev_gate, prev_ts = recent[-2]
                    last_gate, last_ts = recent[-1]

                    if prev_gate == 'top' and last_gate == 'bottom':
                        if filter_obj is not None:
                            if t.is_countable:
                                count_in += 1
                                t.counted = True
                                if args.debug:
                                    print(f"  COUNTED IN Track {t.id} (filtered)")
                            else:
                                if args.debug:
                                    print(f"  SKIPPED IN Track {t.id} (filter mismatch)")
                        elif legacy_white_mode:
                            if t.is_countable:
                                count_in += 1
                                t.counted = True
                                if args.debug:
                                    print(f"  COUNTED IN Track {t.id} (legacy white)")
                            else:
                                if args.debug:
                                    print(f"  SKIPPED IN Track {t.id} (not white)")
                        else:
                            count_in += 1
                            t.counted = True
                            if args.debug:
                                print(f"  COUNTED IN Track {t.id} (unfiltered)")

                    elif prev_gate == 'bottom' and last_gate == 'top':
                        if filter_obj is not None:
                            if t.is_countable:
                                count_out += 1
                                t.counted = True
                                if args.debug:
                                    print(f"  COUNTED OUT Track {t.id} (filtered)")
                            else:
                                if args.debug:
                                    print(f"  SKIPPED OUT Track {t.id} (filter mismatch)")
                        elif legacy_white_mode:
                            if t.is_countable:
                                count_out += 1
                                t.counted = True
                                if args.debug:
                                    print(f"  COUNTED OUT Track {t.id} (legacy white)")
                            else:
                                if args.debug:
                                    print(f"  SKIPPED OUT Track {t.id} (not white)")
                        else:
                            count_out += 1
                            t.counted = True
                            if args.debug:
                                print(f"  COUNTED OUT Track {t.id} (unfiltered)")

        active_ids = set([t.id for t in tracks])
        for tid in list(speed_est.hist.keys()):
            if tid not in active_ids:
                speed_est.remove_track(tid)

        fastest = None
        fastest_v = -1.0
        for t in tracks:
            if getattr(t, 'speed_kmh', None) is not None:
                if t.speed_kmh > fastest_v:
                    fastest_v = t.speed_kmh
                    fastest = t

        for t in tracks:
            if t.class_name == 'car':
                base_color = (0, 255, 0) if t.is_countable else (0, 255, 255)
            elif t.class_name in ('truck','bus','van'):
                base_color = (0, 165, 255)
            elif t.class_name in ('motorbike','motorcycle'):
                base_color = (255, 0, 255)
            else:
                base_color = (200,200,200)

            if fastest is not None and t.id == fastest.id:
                box_color = (0,0,255)  
                thickness = 3
            else:
                box_color = base_color
                thickness = 2

            speed_txt = f"{t.speed_kmh:.1f}km/h" if (getattr(t, 'speed_kmh', None) is not None) else ""
            label = f'ID:{t.id} {t.class_name or ""} {speed_txt}'
            draw_box(frame, t.bbox, label=label, color=box_color, thickness=thickness)
            try:
                cv2.circle(frame, t.centroid, 4, box_color, -1)
            except Exception:
                pass

        mode_desc = args.filter_name if args.filter_name else ('white-only' if legacy_white_mode else 'all')
        cv2.putText(frame, f'Incoming ({mode_desc}): {count_in}', (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f'Outgoing ({mode_desc}): {count_out}', (12,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        now = time.time()
        fps_val = 1.0 / (now - prev_time + 1e-8)
        prev_time = now
        cv2.putText(frame, f'FPS: {fps_val:.1f}', (12,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

        cv2.imshow('output', frame)

        if args.fps_sync:
            elapsed = time.time() - start_frame
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('p'):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    print("Finished. incoming:", count_in, "outgoing:", count_out)

if __name__ == '__main__':
    main()
