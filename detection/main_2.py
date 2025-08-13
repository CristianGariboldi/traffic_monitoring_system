#!/usr/bin/env python3
"""
main_2.py - TWO-GATE (full-width) gate-based counting

Behavior:
 - Two full-width gates: top (green) and bottom (red).
 - If a track enters top then later bottom within seq_window => INCOMING.
 - If a track enters bottom then later top within seq_window => OUTGOING.
 - Toggle white-only counting with --white-only.

Usage example:
python3 main_2.py --source ./data/Video.mp4 --model ./models/yolo11n.onnx \
  --conf 0.25 --auto-calib --calib-frames 40 --debug --fps-sync --white-only
"""

import argparse
import time
import cv2
import numpy as np

from detector_onnx import ONNXDetector as Detector
from color_filter import is_white_car
from tracker import CentroidTracker

ALLOWED_CLASSES = {'car', 'truck', 'bus', 'van', 'motorbike', 'motorcycle'}

def parse_args():
    p = argparse.ArgumentParser(description="Two-gate full-width vehicle counting")
    p.add_argument('--source', '-s', default=0, help='video file or camera index')
    p.add_argument('--model', default='./models/yolo11n.onnx', help='ONNX model path')
    p.add_argument('--conf', type=float, default=0.35, help='detection confidence threshold (0..1)')
    p.add_argument('--debug', action='store_true', help='print debug info')
    p.add_argument('--auto-calib', action='store_true', help='auto-calibrate gate positions from first frames')
    p.add_argument('--calib-frames', type=int, default=40, help='frames for calibration if --auto-calib')
    p.add_argument('--calib-top-q', type=float, default=0.70, help='top quantile for gate placement')
    p.add_argument('--calib-bottom-q', type=float, default=0.80, help='bottom quantile for gate placement')
    p.add_argument('--seq-window', type=float, default=3.0, help='seconds allowed between two gate entries')
    p.add_argument('--fps-sync', action='store_true', help='match display to source FPS')
    p.add_argument('--white-only', action='store_true', help='count only white personal cars when set')
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

    detector = Detector(args.model, input_size=640, providers=['CPUExecutionProvider'], conf_thres=args.conf, debug=args.debug)
    tracker = CentroidTracker(max_missed=12, max_distance=140)

    count_in = 0
    count_out = 0
    seq_window = float(args.seq_window)

    # calibration buffers
    frame_idx = 0
    ys = []
    auto_calib = bool(args.auto_calib)
    calib_frames = int(args.calib_frames)
    top_q = float(args.calib_top_q)
    bottom_q = float(args.calib_bottom_q)

    # default fractions
    # default_top_frac = 0.20
    # default_bottom_frac = 0.45
    default_top_frac = 0.65
    default_bottom_frac = 0.7

    top_y = None
    bottom_y = None

    prev_time = time.time()

    while True:
        start_frame = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]

        # Auto-calibration collection: collect bottom-center y of detections across frames
        if auto_calib and frame_idx < calib_frames:
            dets_tmp = detector.detect(frame)
            for d in dets_tmp:
                cls = str(d['class_name']).lower()
                if cls not in ALLOWED_CLASSES:
                    continue
                x1,y1,x2,y2 = d['bbox']
                cy = int(y2)  # bottom center y
                ys.append(cy)
            frame_idx += 1
            cv2.putText(frame, f'Calibrating: {frame_idx}/{calib_frames}', (12,140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            # draw interim mid lines to guide
            cv2.line(frame, (0, int(H*0.25)), (W-1, int(H*0.25)), (255,255,0), 1)
            cv2.line(frame, (0, int(H*0.5)), (W-1, int(H*0.5)), (255,255,0), 1)
            cv2.imshow('output', frame)
            if args.fps_sync:
                elapsed = time.time() - start_frame
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            continue

        # After collecting samples, compute quantiles if needed
        if auto_calib and top_y is None and frame_idx >= calib_frames:
            if len(ys) >= 4:
                top_y = int(np.quantile(ys, top_q))
                bottom_y = int(np.quantile(ys, bottom_q))
            else:
                top_y = int(H * default_top_frac)
                bottom_y = int(H * default_bottom_frac)
            auto_calib = False
            print(f"Auto-calibrated gates: top_y={top_y}, bottom_y={bottom_y}")

        # Fallback defaults
        if top_y is None:
            top_y = int(H * default_top_frac)
        if bottom_y is None:
            bottom_y = int(H * default_bottom_frac)

        # gate thickness
        gate_half_h = max(8, int(H * 0.02))  # 2% of height or at least 8 px

        # full-width top and bottom gates
        top_gate = (0, max(0, top_y - gate_half_h), W - 1, min(H - 1, top_y + gate_half_h))
        bottom_gate = (0, max(0, bottom_y - gate_half_h), W - 1, min(H - 1, bottom_y + gate_half_h))

        # 1) detect and prepare for tracker
        dets = detector.detect(frame)
        all_dets_for_tracker = []
        for d in dets:
            cls = str(d['class_name']).lower()
            if cls not in ALLOWED_CLASSES:
                continue
            is_personal_car = (cls == 'car')
            is_white = False
            if is_personal_car:
                is_white = is_white_car(frame, d['bbox'])
            is_countable = bool(is_personal_car and is_white)
            all_dets_for_tracker.append({'bbox': d['bbox'], 'class_name': cls, 'is_countable': is_countable})
            if args.debug and is_countable:
                print("DEBUG: detection marked countable:", d['bbox'], "cls=", cls)

        # 2) update tracker
        tracks = tracker.update(all_dets_for_tracker)

        # draw gates
        cv2.rectangle(frame, (top_gate[0], top_gate[1]), (top_gate[2], top_gate[3]), (0,255,0), 2)    # green top
        cv2.rectangle(frame, (bottom_gate[0], bottom_gate[1]), (bottom_gate[2], bottom_gate[3]), (0,0,255), 2)  # red bottom

        now_ts = time.time()

        # iterate tracks -> detect gate entry events and count sequences
        for t in tracks:
            # draw
            if t.class_name == 'car':
                color = (0, 255, 0) if t.is_countable else (0, 255, 255)
            elif t.class_name in ('truck','bus','van'):
                color = (0, 165, 255)
            elif t.class_name in ('motorbike','motorcycle'):
                color = (255, 0, 255)
            else:
                color = (200,200,200)

            label = f'ID:{t.id} {t.class_name or ""}'
            draw_box(frame, t.bbox, label=label, color=color)
            cv2.circle(frame, t.centroid, 4, color, -1)

            if args.debug:
                print(f"TRACK {t.id}: centroid={t.centroid} hits={t.hits} miss={t.miss} countable={t.is_countable} counted={t.counted}")

            # gate entry/exit for two gates
            # TOP gate
            inside_top_now = point_in_rect(t.centroid, top_gate)
            was_inside_top = ('top' in t.inside_gates)
            if inside_top_now and not was_inside_top:
                t.enter_gate('top', now_ts)
                if args.debug:
                    print(f"  Track {t.id} ENTER TOP at {now_ts:.2f}")
            elif not inside_top_now and was_inside_top:
                t.exit_gate('top')
                if args.debug:
                    print(f"  Track {t.id} EXIT TOP at {now_ts:.2f}")

            # BOTTOM gate
            inside_bottom_now = point_in_rect(t.centroid, bottom_gate)
            was_inside_bottom = ('bottom' in t.inside_gates)
            if inside_bottom_now and not was_inside_bottom:
                t.enter_gate('bottom', now_ts)
                if args.debug:
                    print(f"  Track {t.id} ENTER BOTTOM at {now_ts:.2f}")
            elif not inside_bottom_now and was_inside_bottom:
                t.exit_gate('bottom')
                if args.debug:
                    print(f"  Track {t.id} EXIT BOTTOM at {now_ts:.2f}")

            # Evaluate sequences
            if not t.counted:
                recent = [ (g, ts) for g, ts in t.gate_history if (now_ts - ts) <= seq_window ]
                if args.debug and recent:
                    print(f"  Track {t.id} recent gates: {recent}")

                if len(recent) >= 2:
                    prev_gate, prev_ts = recent[-2]
                    last_gate, last_ts = recent[-1]

                    # incoming: top -> bottom
                    if prev_gate == 'top' and last_gate == 'bottom':
                        # count depending on white-only mode
                        if (not args.white_only) or t.is_countable:
                            count_in += 1
                            t.counted = True
                            if args.debug:
                                print(f"  COUNTED IN Track {t.id} (top->bottom) is_countable={t.is_countable}")
                        else:
                            if args.debug:
                                print(f"  SKIPPED IN Track {t.id} (not white)")

                    # outgoing: bottom -> top
                    elif prev_gate == 'bottom' and last_gate == 'top':
                        if (not args.white_only) or t.is_countable:
                            count_out += 1
                            t.counted = True
                            if args.debug:
                                print(f"  COUNTED OUT Track {t.id} (bottom->top) is_countable={t.is_countable}")
                        else:
                            if args.debug:
                                print(f"  SKIPPED OUT Track {t.id} (not white)")

        # overlay
        cv2.putText(frame, f'Incoming ({"white only" if args.white_only else "all"}): {count_in}', (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, f'Outgoing ({"white only" if args.white_only else "all"}): {count_out}', (12,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        now = time.time()
        fps_val = 1.0 / (now - prev_time + 1e-8)
        prev_time = now
        cv2.putText(frame, f'FPS: {fps_val:.1f}', (12,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

        cv2.imshow('output', frame)

        # fps sync
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
