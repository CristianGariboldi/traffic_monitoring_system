#!/usr/bin/env python3
"""
eval_predictor_on_dataset.py

Usage examples:
  # ONNX evaluation (recommended)
  python3 eval_predictor_on_dataset.py --dataset dataset.npz --model predictor.onnx --use-onnx True --vis-n 20 --out results.json

  # PyTorch checkpoint evaluation
  python3 eval_predictor_on_dataset.py --dataset dataset.npz --model checkpoint.pth --use-onnx False --device cpu

Notes:
  - n_obs must match the original training n_obs (dataset was prepared with that).
  - model should have been exported with n_in = n_obs - 1 (deltas length).
"""
import argparse, json, os, math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

# optional: onnxruntime & torch imports done lazily inside PredictorWrapper
try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import torch
except Exception:
    torch = None

# ---------------------
# PredictorWrapper (ONNX or PyTorch)
# ---------------------
import json
class PredictorWrapper:
    def __init__(self, model_path, dataset_npz, use_onnx=True, device='cpu'):
        self.model_path = model_path
        self.device = device
        self.use_onnx = bool(use_onnx) and model_path.endswith('.onnx')

        # load stats from dataset.npz (robust)
        npz = np.load(dataset_npz, allow_pickle=True)
        stats = None
        if 'stats' in npz:
            stats_obj = npz['stats']
            # case 1: plain dict already
            if isinstance(stats_obj, dict):
                stats = stats_obj
            # case 2: numpy 0-d object array containing a dict -> use .item()
            elif isinstance(stats_obj, np.ndarray):
                try:
                    maybe = stats_obj.item()  # often extracts the original dict
                    if isinstance(maybe, dict):
                        stats = maybe
                    else:
                        # maybe a structured array or list-like
                        try:
                            stats = dict(maybe)
                        except Exception:
                            # fallback to tolist()
                            try:
                                stats = stats_obj.tolist()
                                # if tolist returns list of pairs, try dict()
                                if isinstance(stats, list):
                                    try:
                                        stats = dict(stats)
                                    except Exception:
                                        pass
                            except Exception:
                                stats = None
                except Exception:
                    # try tolist() as last resort
                    try:
                        stats = stats_obj.tolist()
                        if isinstance(stats, list):
                            try:
                                stats = dict(stats)
                            except Exception:
                                pass
                    except Exception:
                        stats = None
            # case 3: possibly a JSON string stored in the file
            elif isinstance(stats_obj, (str, bytes)):
                try:
                    stats = json.loads(stats_obj)
                except Exception:
                    stats = None
            else:
                # last attempt: try to convert to python container
                try:
                    stats = dict(stats_obj)
                except Exception:
                    stats = None

        # final check
        if stats is None:
            # helpful debug info for the user
            raise RuntimeError(
                "Failed to parse 'stats' from dataset file '{}'.\n"
                "When creating dataset.npz ensure you saved a dict under 'stats', e.g.:\n"
                "  stats = {'dx_mean': dx_mean.tolist(), 'dx_std': dx_std.tolist()}\n"
                "  np.savez_compressed(..., stats=stats)\n\n"
                "Inspect your dataset with:\n"
                "  import numpy as np; d=np.load('dataset.npz', allow_pickle=True); print(d.files); print(type(d['stats']), getattr(d['stats'], 'shape', None), getattr(d['stats'], 'dtype', None)); print(d['stats'].item())\n"
                .format(dataset_npz)
            )

        # now stats should be a mapping-like with 'dx_mean' and 'dx_std'
        try:
            self.mu = np.array(stats['dx_mean'], dtype=np.float32)
            self.sd = np.array(stats['dx_std'], dtype=np.float32) + 1e-8
        except Exception as e:
            raise RuntimeError(f"Parsed 'stats' object but missing expected keys: {e}. Parsed stats: {stats}")

        # initialize model/session
        if self.use_onnx:
            if ort is None:
                raise RuntimeError("onnxruntime not available in this environment but use_onnx=True")
            providers = ['CPUExecutionProvider']
            self.sess = ort.InferenceSession(model_path, providers=providers)
            inp = self.sess.get_inputs()[0]
            out = self.sess.get_outputs()[0]
            self.in_name = inp.name
            self.out_name = out.name
        else:
            if torch is None:
                raise RuntimeError("PyTorch not available in this environment but use_onnx=False")
            ck = torch.load(model_path, map_location=device)
            from model import TrajectoryTransformer
            args = ck.get('args', {}) or {}
            # try to infer shapes from args or dataset
            n_in = args.get('n_in', None)
            m_pred = args.get('m_pred', None)
            if n_in is None or m_pred is None:
                # infer from dataset
                dX = npz.get('dX', None)
                Yrel = npz.get('Yrel', None)
                if dX is None or Yrel is None:
                    raise RuntimeError("PyTorch checkpoint needs 'n_in' and 'm_pred' in checkpoint args, or dataset must contain 'dX' and 'Yrel' to infer shapes.")
                n_in = dX.shape[1]
                m_pred = Yrel.shape[1]
            d_model = args.get('d_model', 128)
            self.n_in = n_in
            self.m_pred = m_pred
            self.model = TrajectoryTransformer(n_in=n_in, m_pred=m_pred, d_model=d_model)
            self.model.load_state_dict(ck['model_state'])
            self.model.to(device)
            self.model.eval()

    def _prepare_input_from_past(self, past_abs):
        """
        past_abs: (n_obs, 2) absolute pixel coords
        We compute deltas between consecutive past points: shape (n_in, 2) where n_in = n_obs - 1
        then normalize using mu/sd and return shape (1, n_in, 2) float32
        """
        past_abs = np.asarray(past_abs, dtype=np.float32)
        if past_abs.shape[0] < 2:
            # pad minimal
            d = np.zeros((1,1,2), dtype=np.float32)
            d = (d - self.mu.reshape((1,2))) / self.sd.reshape((1,2))
            return d
        d = past_abs[1:] - past_abs[:-1]  # (n_in,2)
        d = (d - self.mu.reshape((1,2))) / self.sd.reshape((1,2))
        return d[np.newaxis, :, :].astype(np.float32)

    def predict(self, past_abs):
        """
        past_abs: (n_obs,2)
        returns pred_abs: (m_pred,2) absolute pixel coords (predicted future positions)
        """
        inp = self._prepare_input_from_past(past_abs)
        if self.use_onnx:
            out = self.sess.run([self.out_name], {self.in_name: inp})[0]  # (1, m_pred, 2)
            pred_deltas = out[0]
        else:
            import torch
            x = torch.from_numpy(inp).to(self.device)
            with torch.no_grad():
                out = self.model(x)
            pred_deltas = out.cpu().numpy()[0]
        # model was trained to output future displacements relative to last observed pos
        last = np.asarray(past_abs[-1], dtype=np.float32).reshape((1,2))
        pred_abs = pred_deltas + last
        return pred_abs

# ---------------------
# Evaluation + visualization routine
# ---------------------
def eval_on_dataset(dataset_path, model_path, use_onnx=True, device='cpu', vis_n=0, vis_dir='vis', save_results='results.json'):
    arr = np.load(dataset_path, allow_pickle=True)
    # dataset created in prepare_dataset: X (N,n_obs,2), Y (N,m_pred,2), dX, Yrel, meta
    X = arr['X']       # absolute past positions (N, n_obs, 2)
    Y = arr['Y']       # absolute future positions (N, m_pred, 2)
    meta = arr.get('meta', None)
    N = X.shape[0]
    n_obs = X.shape[1]
    m_pred = Y.shape[1]

    print(f"Dataset samples: {N}, n_obs={n_obs}, m_pred={m_pred}")

    pw = PredictorWrapper(model_path=model_path, dataset_npz=dataset_path, use_onnx=use_onnx, device=device)

    # accumulate errors per horizon
    errors_by_h = [[] for _ in range(m_pred)]
    all_results = []
    vis_dir = Path(vis_dir)
    if vis_n > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(N), desc="predict"):
        past = X[i]   # (n_obs,2)
        fut = Y[i]    # (m_pred,2)
        meta_i = meta[i] if meta is not None else {}
        try:
            pred = pw.predict(past)  # (m_pred,2)
        except Exception as e:
            print("Predict failed for sample", i, "error:", e)
            continue

        errs = np.linalg.norm(pred - fut, axis=1)  # (m_pred,)
        for h in range(m_pred):
            if not np.isnan(errs[h]):
                errors_by_h[h].append(float(errs[h]))

        res = {
            'sample': int(i),
            'meta': meta_i,
            'errs': [float(x) for x in errs.tolist()],
            'pred': pred.tolist(),
            'gt': fut.tolist()
        }
        all_results.append(res)

        # visualization for first vis_n samples: create simple image plotting points
        if vis_n > 0 and i < vis_n:
            # make a canvas slightly larger than bounding box of past+future
            pts = np.vstack([past, fut, pred])
            minx = int(np.min(pts[:,0]) - 30)
            maxx = int(np.max(pts[:,0]) + 30)
            miny = int(np.min(pts[:,1]) - 30)
            maxy = int(np.max(pts[:,1]) + 30)
            if maxx - minx < 200: maxx = minx + 200
            if maxy - miny < 200: maxy = miny + 200
            W = max(640, maxx-minx)
            H = max(360, maxy-miny)
            img = Image.new('RGB', (W, H), (20,20,20))
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()
            # draw past (white), future GT (green), predicted (cyan)
            for (x,y) in past:
                draw.ellipse((x-minx-3, y-miny-3, x-minx+3, y-miny+3), fill=(255,255,255))
            for (x,y) in fut:
                draw.ellipse((x-minx-3, y-miny-3, x-minx+3, y-miny+3), fill=(0,180,0))
            for (x,y) in pred:
                draw.ellipse((x-minx-3, y-miny-3, x-minx+3, y-miny+3), fill=(200,220,255))
            # labels
            draw.text((8,8), f"sample {i}  meta:{meta_i}", fill=(220,220,220), font=font)
            # per-horizon errors
            for h in range(m_pred):
                px,py = pred[h]
                gx,gy = fut[h]
                draw.text((px-minx+6, py-miny-6), f"h{h+1}:{errs[h]:.1f}px", fill=(200,200,200), font=font)
            img.save(vis_dir / f"sample_{i:05d}.png")

    # compute per-horizon metrics
    maes = []
    rmses = []
    counts = []
    for h in range(m_pred):
        arrh = np.array(errors_by_h[h], dtype=np.float32)
        if arrh.size == 0:
            maes.append(None); rmses.append(None); counts.append(0)
        else:
            maes.append(float(arrh.mean()))
            rmses.append(float(math.sqrt((arrh*arrh).mean())))
            counts.append(int(arrh.size))

    # summary print
    print("Per-horizon metrics (pixels):")
    for h in range(m_pred):
        print(f"h{h+1}: count={counts[h]} MAE={maes[h] if maes[h] is not None else 'na'} RMSE={rmses[h] if rmses[h] is not None else 'na'}")

    # save results json (light)
    if save_results:
        out = {
            'model': model_path,
            'dataset': dataset_path,
            'n_samples': N,
            'n_obs': n_obs,
            'm_pred': m_pred,
            'per_h': [{'h': h+1, 'count': counts[h], 'mae_px': maes[h], 'rmse_px': rmses[h]} for h in range(m_pred)],
            'samples': all_results[:1000]  # limit saving to first 1000 samples to avoid huge files
        }
        with open(save_results, 'w') as fh:
            json.dump(out, fh, indent=2)
        print("Wrote results to", save_results)

    return maes, rmses, counts

# ---------------------
# CLI
# ---------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='dataset.npz', help='dataset from prepare_dataset.py')
    p.add_argument('--model', required=True, help='predictor.onnx or checkpoint.pth')
    p.add_argument('--use-onnx', type=lambda s: s.lower() in ('1','true','yes'), default=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--vis-n', type=int, default=0, help='write PNG visualizations for first N samples (0 = none)')
    p.add_argument('--vis-dir', default='vis')
    p.add_argument('--out', default='pred_results.json')
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    maes, rmses, counts = eval_on_dataset(args.dataset, args.model, use_onnx=args.use_onnx, device=args.device, vis_n=args.vis_n, vis_dir=args.vis_dir, save_results=args.out)
