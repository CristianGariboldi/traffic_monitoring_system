import json
import numpy as np
import cv2
import time
from collections import deque

def compute_homography_from_pairs(img_pts, world_pts):
    """
    img_pts: list of 4+ [x,y] image coordinates
    world_pts: list of corresponding 4+ [X,Y] world coords in meters
    returns H (3x3) mapping image -> world (meters)
    """
    img_pts = np.asarray(img_pts, dtype=np.float32)
    world_pts = np.asarray(world_pts, dtype=np.float32)
    if img_pts.shape[0] < 4 or world_pts.shape[0] < 4:
        raise ValueError("Need at least 4 point correspondences for homography.")
    H, status = cv2.findHomography(img_pts, world_pts, method=cv2.RANSAC)
    return H

class SpeedEstimator:
    def __init__(self, H=None, fallback_m_per_px=None, history_len=6, smooth_alpha=0.6):
        """
        H: 3x3 homography matrix mapping image (px) -> ground (meters).
        fallback_m_per_px: float or None. If provided and H is None, use this approximate scale.
        history_len: number of samples kept per track for smoothing / velocity estimation.
        smooth_alpha: EMA alpha for speed smoothing (0..1). If None, use median smoothing.
        """
        self.H = np.array(H, dtype=np.float64) if H is not None else None
        self.fallback_m_per_px = float(fallback_m_per_px) if fallback_m_per_px is not None else None
        self.history_len = int(history_len)
        self.smooth_alpha = float(smooth_alpha) if smooth_alpha is not None else None

        self.hist = {}
        self.smoothed_speed = {}

    @staticmethod
    def image_to_world_point(H, pt):
        """
        pt: (x,y) in image pixels
        returns (X,Y) in world meters using homography H
        """
        p = np.array([[pt]], dtype=np.float64)  # shape (1,1,2)
        warped = cv2.perspectiveTransform(p, H)  # shape (1,1,2)
        X, Y = float(warped[0,0,0]), float(warped[0,0,1])
        return (X, Y)

    def add_observation(self, track_id, image_pt, ts=None):
        """
        Add observation for track_id.
        image_pt: (x,y) pixel coordinates — bottom center of bbox recommended
        ts: timestamp in seconds (float). If None, uses time.time()
        Returns: speed_kmh (float) if available else None
        """
        if ts is None:
            ts = time.time()
        if track_id not in self.hist:
            self.hist[track_id] = deque(maxlen=self.history_len)
            self.smoothed_speed[track_id] = None

        world_pt = None
        if self.H is not None:
            try:
                world_pt = self.image_to_world_point(self.H, image_pt)
            except Exception:
                world_pt = None

        if world_pt is None and self.fallback_m_per_px is not None:
            sx = image_pt[0] * self.fallback_m_per_px
            sy = image_pt[1] * self.fallback_m_per_px
            world_pt = (sx, sy)

        self.hist[track_id].append((ts, (float(image_pt[0]), float(image_pt[1])), (float(world_pt[0]), float(world_pt[1])) if world_pt is not None else (None, None)))

        speed_m_s = None
        items = list(self.hist[track_id])
        valid = [(t, w) for (t, ip, w) in items if w[0] is not None and w[1] is not None]
        if len(valid) >= 2:
            (t0, (x0,y0)) = valid[-2]
            (t1, (x1,y1)) = valid[-1]
            dt = t1 - t0
            if dt <= 0:
                speed_m_s = 0.0
            else:
                dist = ((x1-x0)**2 + (y1-y0)**2)**0.5
                speed_m_s = dist / dt

        if speed_m_s is not None:
            prev = self.smoothed_speed.get(track_id, None)
            if self.smooth_alpha is None:
                inst = []
                for i in range(1, len(valid)):
                    (ta, (xa,ya)) = valid[i-1]
                    (tb, (xb,yb)) = valid[i]
                    dta = tb - ta
                    if dta <= 0: continue
                    inst.append((( (xb-xa)**2 + (yb-ya)**2)**0.5) / dta)
                if len(inst) > 0:
                    speed_m_s = float(np.median(inst))
                else:
                    speed_m_s = float(speed_m_s)
                self.smoothed_speed[track_id] = speed_m_s
            else:
                if prev is None:
                    self.smoothed_speed[track_id] = float(speed_m_s)
                else:
                    self.smoothed_speed[track_id] = float(self.smooth_alpha * speed_m_s + (1.0 - self.smooth_alpha) * prev)

            return self.smoothed_speed[track_id] * 3.6  # km/h

        return None

    def get_speed_kmh(self, track_id):
        v = self.smoothed_speed.get(track_id, None)
        if v is None:
            return None
        return float(v * 3.6)

    def remove_track(self, track_id):
        if track_id in self.hist:
            del self.hist[track_id]
        if track_id in self.smoothed_speed:
            del self.smoothed_speed[track_id]
