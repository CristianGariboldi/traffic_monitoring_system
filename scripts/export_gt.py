#!/usr/bin/env python3
"""
export_gt.py

Run detector + tracker across a video and write a GT JSON file with per-frame tracked positions.

Usage:
    python3 export_gt.py --source ./data/Video.mp4 --model ./models/yolo11n.onnx --out gt_tracks.json --max-frames 1000 --debug
"""

import argparse
import time
import json
from detector_onnx import ONNXDetector as Detector
from tracker import CentroidTracker
import cv2

ALLOWED_CLASSES = {'car', 'truck', 'bus', 'van', 'motorbike', 'motorcycle'}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source', '-s', default="./data/Video.mp4")
    p.add_argument('--model', default='./models/yolo11s.onnx')
    p.add_argument('--out', default='./data/gt_tracks.json')
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--max-frames', type=int, default=0, help='0 => full video')
    p.add_argument('--debug', action='store_true')
    return p.parse_args()

def frame_record(frame_idx, ts, tracks):
    return {
        'frame_idx': int(frame_idx),
        'ts': float(ts),
        'tracks': [
            {
                'id': int(t.id),
                'bbox': [float(v) for v in t.bbox],
                'centroid': [int(t.centroid[0]), int(t.centroid[1])],
                'class_name': t.class_name
            } for t in tracks
        ]
    }

def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Cannot open source:", args.source)
        return

    detector = Detector(args.model, input_size=640, providers=['CPUExecutionProvider'], conf_thres=args.conf, debug=args.debug)
    tracker = CentroidTracker(max_missed=12, max_distance=140)

    out_frames = []
    maxf = int(args.max_frames) if args.max_frames and args.max_frames > 0 else None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        dets = detector.detect(frame)
        all_dets_for_tracker = []
        for d in dets:
            cls = str(d['class_name']).lower()
            if cls not in ALLOWED_CLASSES:
                continue
            all_dets_for_tracker.append({'bbox': d['bbox'], 'class_name': cls})
        tracks = tracker.update(all_dets_for_tracker)

        video_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        ts = (video_msec/1000.0) if (video_msec and video_msec>0) else time.time()

        rec = frame_record(frame_idx, ts, tracks)
        out_frames.append(rec)

        if args.debug and frame_idx % 50 == 0:
            print(f"[export_gt] frame {frame_idx} tracks={len(tracks)}")

        frame_idx += 1
        if maxf is not None and frame_idx >= maxf:
            break

    # write JSON
    out = {'frames': out_frames}
    with open(args.out, 'w') as fh:
        json.dump(out, fh)
    print(f"Wrote GT tracks to {args.out} (frames={len(out_frames)})")

    cap.release()

if __name__ == '__main__':
    main()
