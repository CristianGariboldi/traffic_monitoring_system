#!/usr/bin/env python3
"""
export_onnx.py: export trained PyTorch predictor to ONNX (dynamic batch & sequence dims)
"""
import torch
import argparse
from model import TrajectoryTransformer

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', default='./data/checkpoint.pth')
    p.add_argument('--out', default='./models/predictor.onnx')
    p.add_argument('--n-in', type=int, default=7)
    p.add_argument('--m-pred', type=int, default=12)
    p.add_argument('--d-model', type=int, default=128)
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu')
    model = TrajectoryTransformer(n_in=args.n_in, m_pred=args.m_pred, d_model=args.d_model)
    model.load_state_dict(ck['model_state'])
    model.eval()

    # dummy input: (1, n_in, 2)
    dummy = torch.randn(1, args.n_in, 2)
    torch.onnx.export(model, dummy, args.out,
                      input_names=['past_deltas'],
                      output_names=['pred_deltas'],
                      dynamic_axes={'past_deltas': {0: 'batch', 1: 'n_in'},
                                    'pred_deltas': {0: 'batch', 1: 'm_pred'}},
                      opset_version=14)
    print("Exported ONNX model to", args.out)

if __name__ == '__main__':
    main()
