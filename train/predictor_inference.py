# # predictor_inference.py
# import numpy as np
# import torch
# import onnxruntime as ort

# class PredictorWrapper:
#     def __init__(self, model_path=None, npz_stats_path='dataset.npz', device='cpu', use_onnx=True):
#         d = np.load(npz_stats_path, allow_pickle=True)
#         stats = d['stats'].item() if 'stats' in d else d['stats']
#         self.mu = np.array(stats['dx_mean'], dtype=np.float32)
#         self.sd = np.array(stats['dx_std'], dtype=np.float32)
#         self.device = device
#         self.use_onnx = bool(use_onnx) and model_path.endswith('.onnx')
#         self.model_path = model_path
#         if self.use_onnx:
#             self.sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
#             self.input_name = self.sess.get_inputs()[0].name
#             self.output_name = self.sess.get_outputs()[0].name
#         else:
#             # load PyTorch
#             ck = torch.load(model_path, map_location=device)
#             # need to recreate model with correct shapes (we infer them from ck['args'] if saved)
#             args = ck.get('args', None)
#             from model import TrajectoryTransformer
#             n_in = args.get('n_in') if args and 'n_in' in args else None
#             m_pred = args.get('m_pred') if args and 'm_pred' in args else None
#             # you should store these or pass them; here we attempt to infer by fallback
#             self.model = TrajectoryTransformer(n_in=n_in, m_pred=m_pred, d_model=args.get('d_model', 128))
#             self.model.load_state_dict(ck['model_state'])
#             self.model.to(device)
#             self.model.eval()

#     def _prepare_input(self, past_centroids):
#         """
#         past_centroids: numpy array shape (n_obs,2) absolute pixel coords.
#         We need to convert to deltas between consecutive frames -> shape (1, n_obs-1, 2), normalized by mu/sd
#         """
#         if past_centroids.shape[0] < 2:
#             # pad with zeros
#             dX = np.zeros((1,1,2), dtype=np.float32)
#         else:
#             d = past_centroids[1:] - past_centroids[:-1]  # (n_obs-1,2)
#             d = (d - self.mu.reshape((1,2))) / (self.sd.reshape((1,2)) + 1e-8)
#             dX = d[np.newaxis, :, :].astype(np.float32)
#         return dX

#     def predict(self, past_centroids):
#         """
#         past_centroids: np.array (n_obs,2)
#         returns predicted future absolute coords (m_pred,2) relative to last observed point -> absolute
#         """
#         dX = self._prepare_input(past_centroids)
#         if self.use_onnx:
#             out = self.sess.run([self.output_name], {self.input_name: dX})[0]  # (1, m, 2)
#             pred_deltas = out[0]  # (m,2)
#         else:
#             import torch
#             x = torch.from_numpy(dX).to(self.device)
#             with torch.no_grad():
#                 out = self.model(x).cpu().numpy()
#             pred_deltas = out[0]
#         # denormalize? our model predicts deltas in normalized space if trained so; in our training we normalized inputs only,
#         # targets were in pixel units (Yrel) so model outputs are in pixel units directly.
#         # convert to absolute by adding last observed point
#         last = past_centroids[-1]
#         preds_abs = pred_deltas + last.reshape((1,2))
#         return preds_abs
