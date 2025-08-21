#!/usr/bin/env python3
"""
prepare_dataset.py

Convert GT JSON (exported by export_gt.py) into supervised training samples.
Each sample is a contiguous subsequence belonging to the same track of length n_obs + m_pred.
We represent positions as pixel centroids. The model will learn to predict future displacements
(dx,dy) relative to the last observed point (or relative to previous frame).
Output is saved as dataset.npz containing:
  X_past: (N, n_obs, 2)
  Y_fut:  (N, m_pred, 2)
  meta: list of dicts with 'track_id' and 'start_frame'
"""

import argparse
import json
import numpy as np
from collections import defaultdict

def load_gt(gt_path):
    with open(gt_path, 'r') as fh:
        j = json.load(fh)
    frames = j.get('frames', [])
    # build mapping track_id -> sorted list of (frame_idx, (cx,cy))
    tracks = defaultdict(list)
    for rec in frames:
        fi = int(rec['frame_idx'])
        for t in rec.get('tracks', []):
            tid = int(t['id'])
            cx, cy = int(round(t['centroid'][0])), int(round(t['centroid'][1]))
            tracks[tid].append((fi, (float(cx), float(cy))))
    # sort times
    for tid, seq in tracks.items():
        seq.sort(key=lambda x: x[0])
    return tracks

def build_samples(tracks, n_obs=8, m_pred=12, min_seq_len=None):
    if min_seq_len is None:
        min_seq_len = n_obs + m_pred
    X = []
    Y = []
    meta = []
    for tid, seq in tracks.items():
        frames = [f for f, _ in seq]
        pts = [p for _, p in seq]
        L = len(seq)
        if L < min_seq_len:
            continue
        # sliding windows where frames are contiguous (no missing frames)
        for s in range(0, L - (n_obs + m_pred) + 1):
            window_frames = frames[s:s + n_obs + m_pred]
            # check contiguous frames
            ok = True
            for i in range(1, len(window_frames)):
                if window_frames[i] != window_frames[i-1] + 1:
                    ok = False
                    break
            if not ok:
                continue
            past_pts = pts[s:s+n_obs]
            fut_pts = pts[s+n_obs:s+n_obs+m_pred]
            X.append(np.array(past_pts, dtype=np.float32))   # shape (n_obs, 2)
            Y.append(np.array(fut_pts, dtype=np.float32))    # shape (m_pred, 2)
            meta.append({'track_id': tid, 'start_frame': window_frames[0]})
    if len(X) == 0:
        raise RuntimeError("No samples found. Try lowering n_obs+m_pred or verify GT continuity.")
    X_arr = np.stack(X, axis=0)
    Y_arr = np.stack(Y, axis=0)
    return X_arr, Y_arr, meta

def compute_relative_deltas(X, Y):
    """
    Convert absolute coordinates to relative displacements.
    We'll use last-observed position as origin:
    input: X (N, n_obs, 2), Y (N, m_pred, 2)
    returns:
      inp_rel: deltas between consecutive past points (n_obs-1, 2) optionally, and/or last-deltas
      For simplicity we will use deltas between consecutive past points as input features.
      output: Y_rel: displacements from last observed point for each future horizon:
        y_rel[h] = Y[h] - X[-1]
    We'll produce model inputs of shape (N, n_obs-1, 2) and targets (N, m_pred, 2)
    """
    last = X[:, -1:, :]  # (N, 1, 2)
    # deltas of past consecutive frames
    dX = X[:, 1:, :] - X[:, :-1, :]  # (N, n_obs-1, 2)
    Yrel = Y - last  # (N, m_pred, 2)
    return dX, Yrel

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gt', default='./data/gt_tracks.json')
    p.add_argument('--out', default='./data/dataset.npz')
    p.add_argument('--n-obs', type=int, default=6)
    p.add_argument('--m-pred', type=int, default=10)
    args = p.parse_args()

    tracks = load_gt(args.gt)
    X, Y, meta = build_samples(tracks, n_obs=args.n_obs, m_pred=args.m_pred)
    dX, Yrel = compute_relative_deltas(X, Y)
    # normalization: compute global scale (std) on deltas to normalize training
    dx_mean = np.mean(dX, axis=(0,1))
    dx_std = np.std(dX, axis=(0,1)) + 1e-6
    # we will store stats for later use in inference
    stats = {'dx_mean': dx_mean.tolist(), 'dx_std': dx_std.tolist()}
    np.savez_compressed(args.out, X=X, Y=Y, dX=dX, Yrel=Yrel, meta=meta, stats=stats)
    print(f"Saved dataset to {args.out} -> samples={X.shape[0]} input_shape={dX.shape} target_shape={Yrel.shape}")
    print("Normalization stats:", stats)

if __name__ == '__main__':
    main()
