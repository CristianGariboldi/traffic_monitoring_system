# eval_baseline.py
import numpy as np, math
from tqdm import tqdm

def baseline_predict(past, m_pred):
    # past: (n_obs,2) absolute
    # simple constant velocity (last delta) extrapolation
    if past.shape[0] < 2:
        last = past[-1]
        return np.tile(last.reshape(1,2), (m_pred,1))
    last_delta = past[-1] - past[-2]   # per-frame displacement (px)
    last_pos = past[-1]
    preds = np.array([ last_pos + (i+1)*last_delta for i in range(m_pred) ], dtype=np.float32)
    return preds

def eval_baseline(dataset_npz):
    d = np.load(dataset_npz, allow_pickle=True)
    X = d['X']   # (N, n_obs, 2)
    Y = d['Y']   # (N, m_pred, 2)
    N = X.shape[0]
    m = Y.shape[1]
    errs_h = [[] for _ in range(m)]
    for i in tqdm(range(N)):
        past = X[i]
        fut = Y[i]
        pred = baseline_predict(past, m)
        errs = np.linalg.norm(pred - fut, axis=1)
        for h in range(m):
            errs_h[h].append(float(errs[h]))
    maes = [ float(np.mean(e)) for e in errs_h ]
    rmses = [ float(math.sqrt(np.mean(np.array(e)**2))) for e in errs_h ]
    for h in range(m):
        print(f"h{h+1}: MAE={maes[h]:.3f} RMSE={rmses[h]:.3f}")
    return maes, rmses

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='./data/dataset.npz')
    args = p.parse_args()
    eval_baseline(args.dataset)
