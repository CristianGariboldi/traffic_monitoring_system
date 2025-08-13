# onnx_debug.py
import cv2, numpy as np, onnxruntime as ort, sys, os
from detector import Detector  # working PyTorch detector
# path to your onnx file (change if necessary)
ONNX_PATH = 'yolov8n.onnx'
IMG_PATH = 'data/output.jpg'  # or use first frame from your video

if not os.path.exists(ONNX_PATH):
    print("ONNX file not found at", ONNX_PATH)
    sys.exit(1)
if not os.path.exists(IMG_PATH):
    # try to extract first frame from your video
    vid = 'data/Video.mp4'
    if os.path.exists(vid):
        vc = cv2.VideoCapture(vid)
        ret, frame = vc.read()
        if not ret:
            print("failed to read frame from video")
            sys.exit(1)
        cv2.imwrite('tmp_frame.jpg', frame)
        IMG_PATH = 'tmp_frame.jpg'
    else:
        print("No test image or video found; place an image at", IMG_PATH)
        sys.exit(1)

print("Loading image:", IMG_PATH)
frame = cv2.imread(IMG_PATH)
h,w = frame.shape[:2]
print("Image shape:", frame.shape)

# 1) Run working Detector (PyTorch) for reference
try:
    det = Detector(model_path='yolov8n.pt', device='cpu', conf_thresh=0.25)
    py_dets = det.detect(frame)
    print("\nPyTorch detector (detector.py) returned", len(py_dets), "detections. Sample (first 5):")
    for i,d in enumerate(py_dets[:5]):
        print(i, d)
except Exception as e:
    print("PyTorch detector failed:", e)

# 2) Load ONNX model and print I/O meta
print("\nLoading ONNX model:", ONNX_PATH)
sess = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
print("ONNX inputs:")
for i in sess.get_inputs():
    print(" ", i.name, i.shape, i.type)
print("ONNX outputs:")
for o in sess.get_outputs():
    print(" ", o.name, o.shape, o.type)

# 3) Preprocess single image (simple letterbox to 640) - match detector_onnx preprocess
def letterbox(im, new_shape=(640,640), color=(114,114,114)):
    shape = im.shape[:2]
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

img, r, (pad_x, pad_y) = letterbox(frame, new_shape=(640,640))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_in = img_rgb.astype('float32')/255.0
img_in = np.transpose(img_in, (2,0,1))[None, ...].astype('float32')

inp_name = sess.get_inputs()[0].name
print("\nInput tensor name:", inp_name, "shape:", img_in.shape, img_in.dtype)

# 4) Run ONNX inference
outs = sess.run(None, {inp_name: img_in})
print("\nONNX returned", len(outs), "output arrays.")
for i,o in enumerate(outs):
    a = np.array(o)
    print(f" output[{i}] shape={a.shape} dtype={a.dtype} min={a.min() if a.size else None} max={a.max() if a.size else None}")
    # print first rows
    if a.size:
        flat = a.reshape(-1, a.shape[-1]) if a.ndim >= 2 else a.reshape(-1,1)
        print("  sample rows:")
        rows = flat[:5]
        print(rows)

print("\nDone. Paste this whole output into the chat so I can adapt the parser to your exact ONNX layout.")
