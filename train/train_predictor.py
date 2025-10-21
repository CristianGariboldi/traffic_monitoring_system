#!/usr/bin/env python3
"""
train_predictor.py

Train the Transformer model on dataset.npz produced by prepare_dataset.py.
Saves the best model to checkpoint.pth
"""

import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from model import TrajectoryTransformer

class TrajDataset(Dataset):
    def __init__(self, npz_path, normalize=True):
        d = np.load(npz_path, allow_pickle=True)
        self.X = d['dX'].astype(np.float32)
        self.Y = d['Yrel'].astype(np.float32)
        self.meta = d.get('meta', None)
        self.stats = d.get('stats', None)
        if normalize and self.stats is not None:
            mu = np.array(self.stats.item()['dx_mean'], dtype=np.float32)
            sd = np.array(self.stats.item()['dx_std'], dtype=np.float32)
            self.X = (self.X - mu.reshape((1,1,2))) / (sd.reshape((1,1,2)) + 1e-8)
            self._norm_mu = mu
            self._norm_sd = sd
        else:
            self._norm_mu = np.zeros(2, dtype=np.float32)
            self._norm_sd = np.ones(2, dtype=np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

def mae_per_horizon(preds, targets):
    errs = np.linalg.norm(preds - targets, axis=2)  # (B,m)
    mae = np.mean(errs, axis=0)
    return mae

def train_epoch(model, loader, opt, device, loss_fn):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        out = model(xb)  
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    all_preds = []
    all_targs = []
    total_loss = 0.0
    loss_fn = nn.MSELoss()
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        total_loss += loss.item() * xb.size(0)
        all_preds.append(pred.cpu().numpy())
        all_targs.append(yb.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    targs = np.concatenate(all_targs, axis=0)
    mae = mae_per_horizon(preds, targs)
    rmse = np.sqrt(np.mean((preds - targs)**2, axis=(0,2)))
    return total_loss / len(loader.dataset), mae, rmse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='./data/dataset.npz')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--val-split', type=float, default=0.1)
    p.add_argument('--out', default='./data/checkpoint.pth')
    args = p.parse_args()

    ds = TrajDataset(args.dataset, normalize=True)
    N = len(ds)
    valN = max(1, int(N * args.val_split))
    trainN = N - valN
    train_ds, val_ds = torch.utils.data.random_split(ds, [trainN, valN])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
    sample_x, _ = ds[0]
    n_in = sample_x.shape[0]
    tmp = np.load(args.dataset, allow_pickle=True)
    m_pred = tmp['Yrel'].shape[1]

    model = TrajectoryTransformer(n_in=n_in, m_pred=m_pred, d_model=args.d_model).to(args.device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val = 1e12
    patience = 6
    wait = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, args.device, loss_fn)
        val_loss, val_mae, val_rmse = eval_model(model, val_loader, args.device)
        print(f"Epoch {epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")
        print(f"  val MAE per horizon: {np.array2string(val_mae, precision=2, separator=',')}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss

            torch.save({'model_state': model.state_dict(), 'args': vars(args)}, args.out)
            print("  Saved best model ->", args.out)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break

    print("Training finished. Best val loss:", best_val)

if __name__ == '__main__':
    main()
