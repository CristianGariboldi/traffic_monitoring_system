# filters.py
"""
Modular filter system for detection -> is_countable decision.
Supports:
 - ColorFilter (white, black, red, blue, etc.) using HSV heuristics
 - ClassFilter (allowlist of class names)
 - AreaFilter (min / max bbox area)
 - CombinedFilter (AND / OR composition)
 - load_filter_from_config(config_path, name) to build from YAML/JSON
"""

import os
import json
import cv2
import numpy as np

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# Try to import the legacy color_filter if present (for reuse)
try:
    from color_filter import is_white_car as legacy_is_white_car
except Exception:
    legacy_is_white_car = None

# -------------------------
# Base filter interface
# -------------------------
class BaseFilter:
    def match(self, frame, bbox, class_name=None):
        """Return True if this detection should be counted."""
        raise NotImplementedError


# -------------------------
# ColorFilter
# -------------------------
class ColorFilter(BaseFilter):
    """
    Color-based filter using HSV pixel statistics and masks.
    color_name: 'white','black','red','blue','gray','silver' etc.
    color_params: optional dict to override thresholds
    """

    # default parameters per named color (HSV rules)
    DEFAULTS = {
        'white': dict(sat_thresh=65, val_thresh=170, min_frac=0.18),
        'black': dict(val_thresh=60, sat_thresh=120, min_frac=0.15),
        'red': dict(hue_ranges=[(0,10),(170,180)], sat_max=200, val_min=60, min_frac=0.08),
        'blue': dict(hue_ranges=[(90,140)], sat_min=80, val_min=50, min_frac=0.06),
        'gray': dict(sat_max=50, val_min=60, val_max=200, min_frac=0.18),
        'silver': dict(sat_max=80, val_min=120, min_frac=0.12)
    }

    def __init__(self, color_name='white', params=None, area_margin=0.08):
        self.color = color_name
        self.params = dict(self.DEFAULTS.get(color_name, {}))
        if params:
            self.params.update(params)
        self.area_margin = float(area_margin)

    def _check_white(self, roi, p):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        s = hsv[...,1].astype(np.uint8)
        v = hsv[...,2].astype(np.uint8)
        sat_thresh = int(p.get('sat_thresh', 65))
        val_thresh = int(p.get('val_thresh', 170))
        mask = (v > val_thresh) & (s < sat_thresh)
        mask = (mask).astype('uint8') * 255
        # small opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        frac = (mask>0).sum() / (mask.size + 1e-8)
        return frac, mask

    def _check_black(self, roi, p):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v = hsv[...,2].astype(np.uint8)
        s = hsv[...,1].astype(np.uint8)
        val_thresh = int(p.get('val_thresh', 60))
        sat_thresh = int(p.get('sat_thresh', 120))
        mask = (v < val_thresh) & (s < sat_thresh)
        mask = mask.astype('uint8') * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        frac = (mask>0).sum() / (mask.size + 1e-8)
        return frac, mask

    def _check_hue_ranges(self, roi, hue_ranges, sat_min=50, val_min=50):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h = hsv[...,0].astype(np.uint8)
        s = hsv[...,1].astype(np.uint8)
        v = hsv[...,2].astype(np.uint8)
        mask_tot = np.zeros_like(h, dtype=np.uint8)
        for (lo, hi) in hue_ranges:
            if lo <= hi:
                m = (h >= lo) & (h <= hi)
            else:
                # wrap-around
                m = (h >= lo) | (h <= hi)
            mask_tot = mask_tot | (m & (s >= sat_min) & (v >= val_min))
        mask = (mask_tot).astype('uint8') * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        frac = (mask>0).sum() / (mask.size + 1e-8)
        return frac, mask

    def match(self, frame, bbox, class_name=None):
        x1,y1,x2,y2 = map(int, bbox)
        h = max(1, y2 - y1)
        w = max(1, x2 - x1)
        # crop center-ish / bottom part to reduce roof/window artifacts
        mh = max(1, int(h * self.area_margin))
        mw = max(1, int(w * self.area_margin))
        cy1 = y1 + mh
        cy2 = y2 - 0  # keep bottom
        cx1 = x1 + mw
        cx2 = x2 - mw
        # bounds
        cy1 = max(0, min(frame.shape[0]-1, cy1))
        cy2 = max(cy1+1, min(frame.shape[0], cy2))
        cx1 = max(0, min(frame.shape[1]-1, cx1))
        cx2 = max(cx1+1, min(frame.shape[1], cx2))
        roi = frame[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            return False

        p = self.params
        if self.color == 'white':
            frac, _ = self._check_white(roi, p)
            return frac >= float(p.get('min_frac', 0.18))
        elif self.color == 'black':
            frac, _ = self._check_black(roi, p)
            return frac >= float(p.get('min_frac', 0.12))
        else:
            # hue-based (red/blue/...)
            hue_ranges = p.get('hue_ranges', None)
            if hue_ranges is None:
                # fallback to using legacy if available and color is 'white'
                if self.color == 'white' and legacy_is_white_car is not None:
                    return legacy_is_white_car(frame, bbox)
                return False
            sat_min = int(p.get('sat_min', 60))
            val_min = int(p.get('val_min', 50))
            frac, _ = self._check_hue_ranges(roi, hue_ranges, sat_min, val_min)
            return frac >= float(p.get('min_frac', 0.06))


# -------------------------
# ClassFilter
# -------------------------
class ClassFilter(BaseFilter):
    def __init__(self, classes):
        self.allowed = set([c.lower() for c in classes])

    def match(self, frame, bbox, class_name=None):
        if class_name is None:
            return False
        return class_name.lower() in self.allowed


# -------------------------
# AreaFilter (bbox area)
# -------------------------
class AreaFilter(BaseFilter):
    def __init__(self, min_area=None, max_area=None):
        self.min_area = float(min_area) if min_area is not None else None
        self.max_area = float(max_area) if max_area is not None else None

    def match(self, frame, bbox, class_name=None):
        x1,y1,x2,y2 = map(int, bbox)
        area = max(0, (x2 - x1) * (y2 - y1))
        if self.min_area is not None and area < self.min_area:
            return False
        if self.max_area is not None and area > self.max_area:
            return False
        return True


# -------------------------
# CombinedFilter
# -------------------------
class CombinedFilter(BaseFilter):
    def __init__(self, filters, mode='and'):
        self.filters = filters
        self.mode = mode.lower()

    def match(self, frame, bbox, class_name=None):
        if self.mode == 'and':
            for f in self.filters:
                if not f.match(frame, bbox, class_name):
                    return False
            return True
        elif self.mode == 'or':
            for f in self.filters:
                if f.match(frame, bbox, class_name):
                    return True
            return False
        else:
            raise ValueError("Unknown mode for CombinedFilter: " + str(self.mode))


# -------------------------
# Config loader
# -------------------------
def _load_config(path):
    if path is None:
        return {}
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r') as f:
        if _HAS_YAML and path.lower().endswith(('.yml', '.yaml')):
            return yaml.safe_load(f)
        else:
            # try json fallback
            try:
                return json.load(f)
            except Exception:
                if _HAS_YAML:
                    return yaml.safe_load(f)
                raise

def build_filter_from_dict(d):
    """
    Construct a BaseFilter from a dict spec.
    Supported keys:
      - type: 'color' | 'class' | 'area' | 'combine'
    For 'color': color_name (e.g. 'white'), params (optional)
    For 'class': classes: [ 'car', 'truck' ]
    For 'area': min_area, max_area
    For 'combine': filters: [ {..}, {..} ], mode: 'and'/'or'
    """
    if d is None:
        return None
    t = d.get('type', 'combine')
    if t == 'color':
        color_name = d.get('color', 'white')
        params = d.get('params', None)
        return ColorFilter(color_name=color_name, params=params, area_margin=d.get('area_margin', 0.08))
    if t == 'class':
        classes = d.get('classes', [])
        return ClassFilter(classes)
    if t == 'area':
        return AreaFilter(min_area=d.get('min_area', None), max_area=d.get('max_area', None))
    if t == 'combine':
        subs = d.get('filters', [])
        subfilters = [build_filter_from_dict(sd) for sd in subs if sd is not None]
        mode = d.get('mode', 'and')
        return CombinedFilter(subfilters, mode=mode)
    raise ValueError("Unknown filter type: " + str(t))

def load_filter_from_config(path, name):
    """
    Read a YAML/JSON file containing named filters.
    Format example (YAML):
      filters:
        white_cars:
          type: combine
          mode: and
          filters:
            - type: class
              classes: ['car']
            - type: color
              color: white
    Returns an instance of BaseFilter for the requested name, or None if name == 'all' or not found.
    """
    cfg = _load_config(path)
    if not cfg:
        return None
    if 'filters' not in cfg:
        # allow top-level dict of named filters
        cfg_filters = cfg
    else:
        cfg_filters = cfg['filters']
    if name is None:
        return None
    if name not in cfg_filters:
        raise KeyError(f"Filter name '{name}' not found in config. Available: {list(cfg_filters.keys())}")
    spec = cfg_filters[name]
    return build_filter_from_dict(spec)