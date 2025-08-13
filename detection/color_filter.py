# color_filter.py
import cv2
import numpy as np

def is_white_car(frame, bbox, sat_thresh=65, val_thresh=170, area_ratio=0.6):
    """
    Heuristic to decide if a car ROI is white.

    Args:
      - frame: full BGR image as numpy array
      - bbox: [x1,y1,x2,y2] (pixel coords)
      - sat_thresh: max mean S allowed for "white" (0-255)
      - val_thresh: min mean V required for "white" (0-255)
      - area_ratio: fraction of bbox height used in cropping (use lower part to avoid sky/roof)
    Returns:
      - True if ROI looks white.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    # crop lower-middle third area to avoid roof + windshield glare
    crop_y1 = y1 + int(h * (1 - area_ratio))
    crop_y2 = y2
    crop_x1 = x1 + int(w * 0.1)
    crop_x2 = x2 - int(w * 0.1)

    # bounds check
    crop_y1 = max(0, crop_y1)
    crop_y2 = max(crop_y1 + 1, crop_y2)
    crop_x1 = max(0, crop_x1)
    crop_x2 = max(crop_x1 + 1, crop_x2)

    roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s_mean = int(np.mean(hsv[...,1]))
    v_mean = int(np.mean(hsv[...,2]))

    # quick morphological mask to ignore tiny specular highlights
    # we compute ratio of pixels with high v but low s
    white_mask = (hsv[...,2] > val_thresh) & (hsv[...,1] < sat_thresh)
    white_ratio = np.sum(white_mask) / (roi.shape[0]*roi.shape[1])

    # decide using combined thresholds
    # require a significant white ratio
    return (v_mean >= val_thresh and s_mean <= (sat_thresh + 30)) or (white_ratio > 0.22)
