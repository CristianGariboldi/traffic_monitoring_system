#!/usr/bin/env python3
"""
main_vlm.py - Integrates YOLO object tracking with the LLaVA-34B VLM.
This version uses a dynamic, semi-transparent text box for VLM output.
"""
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import argparse
import time
import cv2
import numpy as np
import subprocess 

from detector_onnx import ONNXDetector as Detector
from tracker import CentroidTracker

ALLOWED_CLASSES = {'car', 'truck', 'bus', 'van', 'motorbike', 'motorcycle'}


def draw_dynamic_text_box(frame, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, color=(0, 255, 255), thickness=1, max_width=500):
    """
    Draws word-wrapped text with a dynamic, semi-transparent background.
    """
    words = text.split(' ')
    lines = []
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
        if text_width > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line
    lines.append(current_line)

    (text_width, text_height), _ = cv2.getTextSize("A", font, font_scale, thickness)
    line_spacing = text_height + 12
    box_height = len(lines) * line_spacing + 10
    
    longest_line_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        if w > longest_line_width:
            longest_line_width = w
    box_width = longest_line_width + 20

    overlay = frame.copy()
    x, y = org
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), (0, 0, 0), -1)
    
    alpha = 0.6  
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i, line in enumerate(lines):
        line_y = y + (i + 1) * line_spacing - 5
        cv2.putText(frame, line, (x + 10, line_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, line, (x + 10, line_y), font, font_scale, color, thickness, cv2.LINE_AA)


def get_vlm_analysis(frame, vlm_model_path, python_executable):
    try:
        temp_image_path = "/tmp/vlm_frame.jpg"
        cv2.imwrite(temp_image_path, frame)
        
        question = "Describe the traffic situation in this image. Is the traffic flowing smoothly, is it congested, or is there an unusual event like an accident, a stalled vehicle, a pedestrian on the road, or extreme weather conditions?"
        
        script_path = os.path.join(project_root, "scripts/run_single_vlm_inference.py")
        command = [
            python_executable, script_path,
            "--model-path", vlm_model_path,
            "--image-file", temp_image_path,
            "--question", question
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout.strip()

    except subprocess.CalledProcessError as e:
        print(f"Error running VLM subprocess: {e}")
        print(f"VLM Stderr: {e.stderr}")
        return "VLM subprocess failed."
    except Exception as e:
        print(f"Error during VLM analysis: {e}")
        return "VLM analysis failed."

def draw_box(frame, bbox, label=None, color=(0,255,0), thickness=2):
    x1,y1,x2,y2 = map(int, bbox)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, max(0,y1-18)), (x1+tw+6, y1), color, -1)
        cv2.putText(frame, label, (x1+2, max(0,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

def parse_args():
    p = argparse.ArgumentParser(description="VLM-enhanced vehicle tracking and scene analysis")
    p.add_argument('--source', '-s', default="./data/Video7.mp4", help='video file or camera index')
    p.add_argument('--model', default='./models/yolo11n.onnx', help='YOLO ONNX model path')
    p.add_argument('--vlm-model-path', default="/path/to/model", help='Path to the LLaVA-34B model directory')
    p.add_argument('--conf', type=float, default=0.35, help='detection confidence threshold (0..1)')
    p.add_argument('--vlm-interval', type=int, default=150, help='Frames between VLM scene analysis calls (e.g., 150 frames = 5s @ 30fps)')
    return p.parse_args()

def main():
    args = parse_args()
    
    print("Loading YOLO detector...")
    detector = Detector(args.model, input_size=640, providers=['CPUExecutionProvider'], conf_thres=args.conf)
    tracker = CentroidTracker(max_missed=12, max_distance=140)
    print("YOLO detector loaded.")

    python_executable = sys.executable

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Cannot open source:", args.source)
        return

    frame_count = 0
    vlm_status_text = "System Initializing..."
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        h_img, w_img = frame.shape[:2]

        if frame_count % args.vlm_interval == 0:
            print(f"\n--- Frame {frame_count}: Freezing video for VLM Analysis ---")
            analysis_result = get_vlm_analysis(frame, args.vlm_model_path, python_executable)
            vlm_status_text = analysis_result
            print(f"VLM Analysis Result: {vlm_status_text}")
            print("--- VLM Analysis Complete. Resuming video. ---\n")

        dets = detector.detect(frame)
        all_dets_for_tracker = [{'bbox': d['bbox'], 'class_name': str(d['class_name']).lower()} for d in dets if str(d['class_name']).lower() in ALLOWED_CLASSES]
        tracks = tracker.update(all_dets_for_tracker)
        
        for t in tracks:
            draw_box(frame, t.bbox, label=f'ID:{t.id}', color=(0, 255, 0))
            cv2.circle(frame, t.centroid, 4, (0, 255, 0), -1)
            
        draw_dynamic_text_box(
            frame, 
            f"Scene Analysis: {vlm_status_text}", 
            org=(15, 15), 
            max_width=500  
        )
        
        now = time.time()
        fps_val = 1.0 / (now - prev_time + 1e-8)
        prev_time = now
        cv2.putText(frame, f'FPS: {fps_val:.1f}', (15, h_img - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('VLM Scene Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()