# tracker.py
import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque

class Track:
    def __init__(self, tid, bbox, class_name=None, is_countable=False, max_history=30):
        self.id = tid
        self.bbox = bbox  # [x1,y1,x2,y2]
        self.class_name = class_name
        self.is_countable = is_countable  
        self.centroid = self._bottom_center(bbox)
        self.hits = 1
        self.miss = 0
        self.counted = False
        self.history = deque(maxlen=max_history)  
        self.history.append(self.centroid)
        self.inside_gates = set()                 
        self.gate_history = deque(maxlen=50)      

    def _bottom_center(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        return (cx, int(y2))

    def update(self, bbox, class_name=None, is_countable=None):
        self.bbox = bbox
        self.centroid = self._bottom_center(bbox)
        if class_name is not None:
            self.class_name = class_name
        if is_countable is not None:
            self.is_countable = bool(self.is_countable or is_countable)
        self.hits += 1
        self.miss = 0
        self.history.append(self.centroid)

    def mark_missed(self):
        self.miss += 1

    def last_two_centroids(self):
        if len(self.history) >= 2:
            return (self.history[-2], self.history[-1])
        elif len(self.history) == 1:
            return (self.history[-1], self.history[-1])
        else:
            return None

    def enter_gate(self, gate_id, timestamp):
        """
        Called when track centroid enters a gate (from outside to inside).
        We record the event only on entry.
        """
        if gate_id in self.inside_gates:
            return
        self.inside_gates.add(gate_id)
        self.gate_history.append((gate_id, timestamp))

    def exit_gate(self, gate_id):
        """
        Called when track centroid exits a gate.
        We simply update inside_gates.
        """
        if gate_id in self.inside_gates:
            self.inside_gates.remove(gate_id)


class CentroidTracker:
    def __init__(self, max_missed=10, max_distance=120):
        self.next_id = 1
        self.tracks = dict()  # id -> Track
        self.max_missed = max_missed
        self.max_distance = max_distance

    def _compute_cost(self, track_centroids, det_centroids):
        t = np.array(track_centroids, dtype=float)
        d = np.array(det_centroids, dtype=float)
        if t.size == 0 or d.size == 0:
            return np.zeros((t.shape[0], d.shape[0]), dtype=float)
        if t.ndim == 1:
            t = t.reshape(1, -1)
        if d.ndim == 1:
            d = d.reshape(1, -1)
        if t.shape[1] != 2:
            t = t.reshape(-1, 2)
        if d.shape[1] != 2:
            d = d.reshape(-1, 2)
        diff = t[:, None, :] - d[None, :, :]  # shape (T, D, 2)
        cost = np.linalg.norm(diff, axis=2)
        return cost

    def update(self, detections):
        """
        detections: list of dicts with keys:
            'bbox': [x1,y1,x2,y2],
            optionally 'class_name': str,
            optionally 'is_countable': bool
        Returns list of Track objects.
        """
        det_bboxes = [d['bbox'] for d in detections]
        det_centroids = [ (int((b[0]+b[2])/2), int(b[3])) for b in det_bboxes ]  # bottom-center
        det_classes = [ d.get('class_name', None) for d in detections ]
        det_countable = [ bool(d.get('is_countable', False)) for d in detections ]

        track_ids = list(self.tracks.keys())
        track_centroids = [ self.tracks[tid].centroid for tid in track_ids ]

        if len(track_centroids) == 0:
            for bbox, cls, cnt in zip(det_bboxes, det_classes, det_countable):
                self.tracks[self.next_id] = Track(self.next_id, bbox, class_name=cls, is_countable=cnt)
                self.next_id += 1
            return list(self.tracks.values())

        if len(det_centroids) == 0:
            for tid in list(track_ids):
                if tid in self.tracks:
                    self.tracks[tid].mark_missed()
                    if self.tracks[tid].miss > self.max_missed:
                        del self.tracks[tid]
            return list(self.tracks.values())

        cost = self._compute_cost(track_centroids, det_centroids)
        row_ind, col_ind = linear_sum_assignment(cost)

        assigned_tracks = set()
        assigned_dets = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= self.max_distance:
                tid = track_ids[r]
                bbox = det_bboxes[c]
                cls = det_classes[c]
                cnt = det_countable[c]
                if tid in self.tracks:
                    self.tracks[tid].update(bbox, class_name=cls, is_countable=cnt)
                    assigned_tracks.add(tid)
                    assigned_dets.add(c)

        for i, bbox in enumerate(det_bboxes):
            if i not in assigned_dets:
                self.tracks[self.next_id] = Track(self.next_id, bbox,
                                                 class_name=det_classes[i],
                                                 is_countable=det_countable[i])
                self.next_id += 1

        for tid in list(track_ids):
            if tid not in assigned_tracks:
                if tid in self.tracks:
                    self.tracks[tid].mark_missed()
                    if self.tracks[tid].miss > self.max_missed:
                        del self.tracks[tid]

        return list(self.tracks.values())

    @staticmethod
    def point_line_sign(pt, line_p1, line_p2):
        x, y = pt
        x1, y1 = line_p1
        x2, y2 = line_p2
        return (x - x1)*(y2 - y1) - (y - y1)*(x2 - x1)
