#!/usr/bin/env python3
"""
main_2.py - TWO-GATE counting with config-driven filters

Usage examples:
  # count all vehicles (no config)
  python3 main_2.py --source ./data/Video.mp4 --model ./models/yolo11n.onnx --conf 0.25

  # use config file + filter name (count only white cars)
  python3 main_2.py --source ./data/Video.mp4 --model ./models/yolo11n.onnx \
      --conf 0.25 --filter-config filters.yaml --filter-name white_cars --debug

"""

import argparse
import time
import cv2
import numpy as np
from detector_onnx import ONNXDetector as Detector
from tracker import CentroidTracker

# Try to import the filters loader
from filters import load_filter_from_config
# Legacy color function fallback if you kept color_filter.py
try:
    from color_filter import is_white_car as legacy_is_white_car
except Exception:
    legacy_is_white_car = None

ALLOWED_CLASSES = {'car', 'truck', 'bus', 'van', 'motorbike', 'motorcycle'}

def parse_args():
    p = argparse.ArgumentParser(description="Two-gate counting with config-driven filters")
    p.add_argument('--source', '-s', default=0, help='video file or camera index')
    p.add_argument('--model', default='./models/yolo11n.onnx', help='ONNX model path')
    p.add_argument('--conf', type=float, default=0.35, help='detection confidence threshold (0..1)')
    p.add_argument('--debug', action='store_true', help='print debug info')
    p.add_argument('--filter-config', default=None, help='path to filters YAML/JSON config')
    p.add_argument('--filter-name', default=None, help='named filter in config (if omitted, no filter => count all)')
    p.add_argument('--seq-window', type=float, default=3.0, help='seconds allowed between two gate entries')
    p.add_argument('--fps-sync', action='store_true', help='match display to source FPS')
    # keep --white-only for backwards compatibility
    p.add_argument('--white-only', action='store_true', help='legacy: if set, count only white cars using old heuristic')
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

    # load filter if requested
    filter_obj = None
    if args.filter_config and args.filter_name:
        try:
            filter_obj = load_filter_from_config(args.filter_config, args.filter_name)
            print(f"Loaded filter '{args.filter_name}' from {args.filter_config}")
        except Exception as e:
            print("Failed to load filter config:", e)
            filter_obj = None

    # legacy white-only fallback (if user used --white-only)
    legacy_white_mode = bool(args.white_only) and (filter_obj is None)

    count_in = 0
    count_out = 0
    seq_window = float(args.seq_window)

    # auto-calib style is kept simple: you can re-use your auto-calib logic if desired.
    # here we use sensible defaults
    default_top_frac = 0.65
    default_bottom_frac = 0.70

    top_y = None
    bottom_y = None

    prev_time = time.time()

    while True:
        start_frame = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]

        # set gate positions (simple defaults)
        if top_y is None:
            top_y = int(H * default_top_frac)
        if bottom_y is None:
            bottom_y = int(H * default_bottom_frac)

        gate_half_h = max(8, int(H * 0.02))
        top_gate = (0, max(0, top_y - gate_half_h), W - 1, min(H - 1, top_y + gate_half_h))
        bottom_gate = (0, max(0, bottom_y - gate_half_h), W - 1, min(H - 1, bottom_y + gate_half_h))

        dets = detector.detect(frame)
        all_dets_for_tracker = []
        for d in dets:
            cls = str(d['class_name']).lower()
            if cls not in ALLOWED_CLASSES:
                continue

            # Decision: compute is_countable using filter_obj if present.
            is_countable = True
            if filter_obj:
                try:
                    is_countable = bool(filter_obj.match(frame, d['bbox'], cls))
                except Exception:
                    is_countable = False
            elif legacy_white_mode:
                # fallback to legacy color function if available
                if legacy_is_white_car is not None and cls == 'car':
                    is_countable = bool(legacy_is_white_car(frame, d['bbox']))
                else:
                    is_countable = False
            else:
                # no filter - count all
                is_countable = True

            all_dets_for_tracker.append({'bbox': d['bbox'], 'class_name': cls, 'is_countable': is_countable})
            if args.debug and not is_countable:
                # helpful to know why a detection is skipped
                print(f"DEBUG not countable: cls={cls}, bbox={d['bbox']} is_countable={is_countable}")

        tracks = tracker.update(all_dets_for_tracker)

        # draw gates
        cv2.rectangle(frame, (top_gate[0], top_gate[1]), (top_gate[2], top_gate[3]), (0,255,0), 2)    # green top
        cv2.rectangle(frame, (bottom_gate[0], bottom_gate[1]), (bottom_gate[2], bottom_gate[3]), (0,0,255), 2)  # red bottom

        now_ts = time.time()

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
                print(f"TRACK {t.id}: centroid={t.centroid} hits={t.hits} miss={t.miss} is_countable={t.is_countable} counted={t.counted}")

            # gate entry/exit logic (two full-width gates)
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

            # evaluate sequences
            if not t.counted:
                recent = [ (g, ts) for g, ts in t.gate_history if (now_ts - ts) <= seq_window ]
                if args.debug and recent:
                    print(f"  Track {t.id} recent gates: {recent}")

                if len(recent) >= 2:
                    prev_gate, prev_ts = recent[-2]
                    last_gate, last_ts = recent[-1]

                    # incoming top->bottom
                    if prev_gate == 'top' and last_gate == 'bottom':
                        # if a filter is configured, only count if track matched it (t.is_countable True)
                        if filter_obj:
                            if t.is_countable:
                                count_in += 1
                                t.counted = True
                                if args.debug:
                                    print(f"  COUNTED IN Track {t.id} (filtered)")
                        else:
                            # if no filter configured, count all (legacy behavior)
                            count_in += 1
                            t.counted = True
                            if args.debug:
                                print(f"  COUNTED IN Track {t.id} (unfiltered)")

                    # outgoing bottom->top
                    elif prev_gate == 'bottom' and last_gate == 'top':
                        if filter_obj:
                            if t.is_countable:
                                count_out += 1
                                t.counted = True
                                if args.debug:
                                    print(f"  COUNTED OUT Track {t.id} (filtered)")
                        else:
                            count_out += 1
                            t.counted = True
                            if args.debug:
                                print(f"  COUNTED OUT Track {t.id} (unfiltered)")

        # overlay
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
