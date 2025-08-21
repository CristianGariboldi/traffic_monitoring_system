#!/usr/bin/env python3
"""
predict_eval_cleanvis.py (updated fixes)

Cleaner visualization for prediction evaluation:
 - predicted future positions: cyan dots (shown only while prediction horizons are still future)
 - GT future positions: green dots (if GT exists)
 - bbox label includes ID and avg prediction error for most recent prediction for that track (if available)
 - supports two modes: 'gt' (recommended) and 'live' (run detector+tracker live)
 
Fixes in this version:
 - prediction records are purged per-record once their horizons expire
 - in live mode predictions are evaluated immediately against GT if GT has entries for the future frames
"""

import argparse
import json
import time
import math
from collections import defaultdict, deque
import numpy as np
import cv2

# reuse CentroidTracker and ONNXDetector
from tracker import CentroidTracker
from detector_onnx import ONNXDetector as Detector

# --- lightweight Kalman2D (same as earlier) ---
class Kalman2D:
    def __init__(self, x=0.0, y=0.0, vx=0.0, vy=0.0, dt=1/25.0,
                 proc_var_pos=1.0, proc_var_vel=1.0, meas_var=4.0):
        self.x = np.array([[x],[y],[vx],[vy]], dtype=float)
        self.dt = float(dt)
        self.F = np.array([[1,0,self.dt,0],[0,1,0,self.dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.Q = np.diag([proc_var_pos, proc_var_pos, proc_var_vel, proc_var_vel]).astype(float)
        self.R = np.eye(2, dtype=float) * float(meas_var)
        self.P = np.eye(4, dtype=float) * 100.0
        self._I = np.eye(4, dtype=float)

    def predict(self):
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
        return float(self.x[0,0]), float(self.x[1,0])

    def update(self, px, py):
        z = np.array([[px],[py]], dtype=float)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S + 1e-8 * np.eye(2)))
        y = z - self.H.dot(self.x)
        self.x = self.x + K.dot(y)
        self.P = (self._I - K.dot(self.H)).dot(self.P)

    def current_position(self):
        return float(self.x[0,0]), float(self.x[1,0])

    def clone(self):
        k = Kalman2D()
        k.x = self.x.copy()
        k.P = self.P.copy()
        k.F = self.F.copy()
        k.H = self.H.copy()
        k.Q = self.Q.copy()
        k.R = self.R.copy()
        k._I = self._I.copy()
        k.dt = self.dt
        return k

# --- PredictionManager with per-record error storage + purge logic ---
class PredictionManager:
    def __init__(self, fps=25.0, n_obs=4, m_pred=8, predict_interval=6, debug=False, purge_margin=0):
        self.fps = float(fps)
        self.dt = 1.0/self.fps
        self.n_obs = int(n_obs)
        self.m_pred = int(m_pred)
        self.predict_interval = int(predict_interval)
        self.debug = bool(debug)

        self.kfs = {}  # per-id Kalman
        self.hist = defaultdict(lambda: deque(maxlen=200))  # id -> deque((frame_idx, ts, (x,y)))
        # pred_records: id -> list of events; each event has:
        # {'frame_idx', 'ts', 'predictions':[(x,y)...], 'errors':[err or None,...], 'avg_error':float or None}
        self.pred_records = defaultdict(list)
        self.errors_by_horizon = defaultdict(list)
        self.pred_count = 0
        self.purge_margin = int(purge_margin)

    def update_observation(self, tid, cx, cy, frame_idx, ts):
        self.hist[tid].append((frame_idx, ts, (float(cx), float(cy))))
        if tid not in self.kfs:
            vx = vy = 0.0
            if len(self.hist[tid]) >= 2:
                (f0,t0,(x0,y0)), (f1,t1,(x1,y1)) = self.hist[tid][-2], self.hist[tid][-1]
                dt = max(1e-6, t1 - t0)
                vx = (x1 - x0) / dt
                vy = (y1 - y0) / dt
            self.kfs[tid] = Kalman2D(x=cx, y=cy, vx=vx, vy=vy, dt=self.dt, proc_var_pos=2.0, proc_var_vel=1.0, meas_var=4.0)
            if self.debug:
                print(f"[PM] init KF id={tid} pos=({cx:.1f},{cy:.1f}) vel=({vx:.2f},{vy:.2f})")
        else:
            kf = self.kfs[tid]
            kf.predict()
            kf.update(float(cx), float(cy))

    def maybe_predict(self, tid, frame_idx, ts, gt_by_frame=None):
        """
        Possibly create a prediction for 'tid' at frame_idx.
        If gt_by_frame is provided we immediately compute per-horizon errors and store them in the record.
        """
        made = False
        if len(self.hist[tid]) >= self.n_obs:
            last_pred_frame = self.pred_records[tid][-1]['frame_idx'] if self.pred_records[tid] else -999999
            if (frame_idx - last_pred_frame) >= self.predict_interval:
                kf_copy = self.kfs[tid].clone()
                preds = []
                for s in range(1, self.m_pred+1):
                    kf_copy.predict()
                    px,py = kf_copy.current_position()
                    preds.append((float(px), float(py)))
                rec = {'frame_idx': frame_idx, 'ts': ts, 'predictions': preds, 'errors': [None]*len(preds), 'avg_error': None}
                # if gt_by_frame provided evaluate right away (works for live mode too)
                if gt_by_frame is not None:
                    errs = []
                    for h_idx, (px,py) in enumerate(preds, start=1):
                        gt_frame = frame_idx + h_idx
                        frame_gt = gt_by_frame.get(gt_frame)
                        if frame_gt:
                            info = frame_gt.get(str(tid))
                            if info:
                                gx, gy = info['centroid']
                                err = math.hypot(px - gx, py - gy)
                                rec['errors'][h_idx-1] = float(err)
                                errs.append(err)
                                self.errors_by_horizon[h_idx].append(err)
                    if len(errs) > 0:
                        rec['avg_error'] = float(sum(errs)/len(errs))
                self.pred_records[tid].append(rec)
                self.pred_count += 1
                made = True
                if self.debug:
                    print(f"[PM] predicted id={tid} at frame {frame_idx}; avg_err={rec['avg_error']}")
        return made

    def purge_expired_records(self, current_frame):
        """
        Remove any prediction record that is older than start + m_pred + purge_margin
        so they do not linger forever (fixes visualization clutter).
        """
        cutoff = current_frame - (self.m_pred + self.purge_margin)
        for tid, recs in list(self.pred_records.items()):
            new_recs = []
            for rec in recs:
                if rec['frame_idx'] >= cutoff:
                    new_recs.append(rec)
            if new_recs:
                self.pred_records[tid] = new_recs
            else:
                # remove key entirely if no recs left
                del self.pred_records[tid]

    def cleanup_track(self, tid):
        if tid in self.kfs: del self.kfs[tid]
        if tid in self.hist: del self.hist[tid]
        # keep pred_records for possible visualization until they expire

    def metrics(self):
        out = {}
        for h in range(1, self.m_pred+1):
            arr = self.errors_by_horizon.get(h, [])
            if len(arr) == 0:
                out[h] = {'count':0, 'mae_px':None, 'rmse_px':None}
            else:
                a = np.array(arr, dtype=float)
                out[h] = {'count': int(a.size), 'mae_px': float(a.mean()), 'rmse_px': float(math.sqrt((a*a).mean()))}
        return out

# --- load GT file utils ---
def load_gt(gt_path):
    with open(gt_path, 'r') as fh:
        j = json.load(fh)
    frames = j.get('frames', [])
    gt_by_frame = {}
    for rec in frames:
        fi = int(rec['frame_idx'])
        mapping = {}
        for t in rec.get('tracks', []):
            mapping[str(int(t['id']))] = {'bbox': t['bbox'], 'centroid': [float(t['centroid'][0]), float(t['centroid'][1])], 'class_name': t.get('class_name')}
        gt_by_frame[fi] = mapping
    return frames, gt_by_frame

# --- main script args ---
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gt', default="./data/gt_tracks.json", help='GT file path (from export_gt.py)')
    p.add_argument('--mode', choices=['gt', 'live'], default='gt', help='gt: feed GT positions to predictor. live: run detector+tracker live.')
    p.add_argument('--source', default='./data/Video.mp4', help='video (used in live mode)')
    p.add_argument('--model', default='./models/yolo11n.onnx', help='detection model (live mode)')
    p.add_argument('--conf', type=float, default=0.35)
    p.add_argument('--n-obs', type=int, default=6)
    p.add_argument('--m-pred', type=int, default=10)
    p.add_argument('--predict-interval', type=int, default=6)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--fps-sync', action='store_true')
    p.add_argument('--purge-margin', type=int, default=0, help='extra frames to keep records beyond m_pred (default 0)')
    return p.parse_args()

def main():
    args = parse_args()
    frames_list, gt_by_frame = load_gt(args.gt)
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Cannot open source:", args.source)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dt = 1.0 / fps

    # live mode detector/tracker
    if args.mode == 'live':
        detector = Detector(args.model, input_size=640, providers=['CPUExecutionProvider'], conf_thres=args.conf, debug=args.debug)
        tracker = CentroidTracker(max_missed=12, max_distance=140)

    pm = PredictionManager(fps=fps, n_obs=args.n_obs, m_pred=args.m_pred, predict_interval=args.predict_interval, debug=args.debug, purge_margin=args.purge_margin)

    frame_idx = 0
    # we will show only recent records -> but they are purged more aggressively now
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h_img, w_img = frame.shape[:2]
        video_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        ts = (video_msec / 1000.0) if (video_msec and video_msec>0) else time.time()

        if args.mode == 'live':
            dets = detector.detect(frame)
            all_dets = []
            for d in dets:
                cls = str(d['class_name']).lower()
                if cls not in {'car','truck','bus','van','motorbike','motorcycle'}:
                    continue
                all_dets.append({'bbox': d['bbox'], 'class_name': cls})
            tracks = tracker.update(all_dets)
            obs_tracks = tracks
        else:
            # GT mode: iterate GT track entries
            mapping = gt_by_frame.get(frame_idx, {})
            obs_tracks = []
            for sid, info in mapping.items():
                tid = int(sid)
                bbox = info['bbox']
                cx, cy = info['centroid']
                class_name = info.get('class_name', 'car')
                tr = type('T', (), {})()
                tr.id = tid
                tr.bbox = bbox
                tr.class_name = class_name
                tr.centroid = (int(round(cx)), int(round(cy)))
                obs_tracks.append(tr)

        # update PM with observations and possibly make predictions
        for t in obs_tracks:
            pm.update_observation(t.id, t.centroid[0], t.centroid[1], frame_idx, ts)
            # now pass gt_by_frame even in live mode so immediate evaluation can happen if GT exists
            pm.maybe_predict(t.id, frame_idx, ts, gt_by_frame=gt_by_frame)

        # cleanup PM tracks not seen now
        active_ids = set([t.id for t in obs_tracks])
        for tid in list(pm.kfs.keys()):
            if tid not in active_ids:
                pm.cleanup_track(tid)

        # purge expired prediction records so they don't linger after their horizons pass
        pm.purge_expired_records(frame_idx)

        # Drawing:
        # 1) draw current detections/tracks and bbox label with accuracy (most recent rec)
        for t in obs_tracks:
            x1,y1,x2,y2 = map(int, t.bbox)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,200,0), 2)
            # compute accuracy text using most recent prediction record for this track (if any)
            recent_recs = pm.pred_records.get(t.id, [])
            acc_txt = "e=-"
            if recent_recs:
                last_rec = recent_recs[-1]
                if last_rec.get('avg_error') is not None:
                    acc_txt = f"e={last_rec['avg_error']:.1f}px"
                else:
                    # if no avg_error but h1 error exists, show h1
                    if len(last_rec.get('errors', [])) >= 1 and last_rec['errors'][0] is not None:
                        acc_txt = f"h1={last_rec['errors'][0]:.1f}px"
                    else:
                        acc_txt = "e=-"
            label = f"ID:{t.id} {acc_txt}"
            cv2.putText(frame, label, (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,0), 1)
            cv2.circle(frame, t.centroid, 3, (0,200,0), -1)

        # 2) draw predicted vs GT future points (no lines)
        # Only draw prediction records that are still within their horizon window (they are purged otherwise)
        for tid, recs in pm.pred_records.items():
            for rec in recs:
                start = rec['frame_idx']
                # only show predictions if current frame is between start (inclusive) and start + m_pred (inclusive)
                if not (frame_idx >= start and frame_idx <= start + pm.m_pred):
                    continue
                preds = rec['predictions']
                # draw predicted trajectory dots (cyan)
                for (px,py) in preds:
                    p_pt = (int(round(px)), int(round(py)))
                    cv2.circle(frame, p_pt, 3, (255,200,0), -1)  # cyan
                # draw GT future points (green) for horizons where GT exists
                for h_idx, (px,py) in enumerate(preds, start=1):
                    gt_frame = start + h_idx
                    frame_gt = gt_by_frame.get(gt_frame)
                    if frame_gt:
                        info = frame_gt.get(str(tid))
                        if info:
                            gx, gy = info['centroid']
                            g_pt = (int(round(gx)), int(round(gy)))
                            cv2.circle(frame, g_pt, 3, (0,255,0), -1)  # green

        # 3) overlay aggregated metrics (compact)
        metrics = pm.metrics()
        y0 = 18
        cv2.putText(frame, f"PredCount:{pm.pred_count}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        # only print first 6 horizons to not overcrowd
        max_show = min(6, pm.m_pred)
        for h in range(1, max_show+1):
            m = metrics[h]
            if m['count'] > 0:
                txt = f"h{h}: mae={m['mae_px']:.1f}px rmse={m['rmse_px']:.1f}px"
            else:
                txt = f"h{h}: no-data"
            cv2.putText(frame, txt, (10, y0 + 20*h), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)

        cv2.imshow('predict_eval_clean', frame)
        if args.fps_sync:
            time.sleep(dt)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        frame_idx += 1

    # End: print final metrics
    print("Final metrics (pixels):")
    final = pm.metrics()
    for h in range(1, pm.m_pred+1):
        m = final[h]
        if m['count'] > 0:
            print(f"h={h}: count={m['count']} MAE={m['mae_px']:.2f}px RMSE={m['rmse_px']:.2f}px")
        else:
            print(f"h={h}: count=0")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
