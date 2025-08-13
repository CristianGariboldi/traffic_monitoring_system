# detector_onnx.py
"""
Robust ONNXRuntime detector wrapper for Ultralytics-style YOLO ONNX exports.
Replace the previous file with this one.

Usage:
    det = ONNXDetector('models/yolov8n.onnx', input_size=640, providers=['CPUExecutionProvider'], conf_thres=0.25, debug=True)
    det.detect(frame) -> list of dicts { 'bbox':[x1,y1,x2,y2], 'conf': float, 'class_name': str }

Notes:
 - This wrapper auto-detects several ONNX output formats and adapts.
 - If no detections appear, enable debug=True and run a single frame to inspect the raw output.
"""

import os
import numpy as np
import onnxruntime as ort
import cv2

# COCO names fallback
COCO_NAMES = {i: n for i, n in enumerate([
    'person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','sofa',
    'pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone',
    'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
    'hair drier','toothbrush'])}

# -------------------------
# utility helpers
# -------------------------
def letterbox(im, new_shape=(640, 640), color=(114,114,114), scaleUp=True):
    # preserve aspect ratio, pad to new_shape
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleUp:
        r = min(r, 1.0)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    # resize
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def nms_simple(boxes, scores, iou_thres=0.45):
    # simple NMS, returns indices to keep
    if boxes.shape[0] == 0:
        return []
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = _bbox_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious <= iou_thres]
    return keep

def _bbox_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:,0])
    y1 = np.maximum(box[1], boxes[:,1])
    x2 = np.minimum(box[2], boxes[:,2])
    y2 = np.minimum(box[3], boxes[:,3])
    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter = inter_w * inter_h
    area1 = max(0, (box[2]-box[0]) * (box[3]-box[1]))
    area2 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

# -------------------------
# ONNX Detector class
# -------------------------
class ONNXDetector:
    def __init__(self, onnx_path, input_size=640, providers=None, conf_thres=0.25, iou_thres=0.45, class_names=None, debug=False):
        assert os.path.exists(onnx_path), f"ONNX model not found: {onnx_path}"
        self.onnx_path = onnx_path
        self.input_size = int(input_size)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.debug = bool(debug)

        if providers is None:
            providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # class names
        if class_names is None:
            self.class_names = COCO_NAMES
        else:
            self.class_names = class_names

        if self.debug:
            print(f"[ONNXDetector] loaded {onnx_path}")
            print(f" ONNX inputs: {[ (i.name, i.shape, i.type) for i in self.session.get_inputs() ]}")
            print(f" ONNX outputs: {[ (o.name, o.shape, o.type) for o in self.session.get_outputs() ]}")

    def preprocess(self, frame):
        img, ratio, (pad_x, pad_y) = letterbox(frame, new_shape=(self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))
        img = np.expand_dims(img, 0).astype(np.float32)
        return img, ratio, pad_x, pad_y

    def _interpret_and_convert(self, preds, ratio, pad_x, pad_y, orig_shape):
        """
        Convert raw preds tensor into list of {bbox, conf, class_id, class_name}
        Handles common ONNX layouts:
         - (1, C, N) -> transpose to (1, N, C) when C << N
         - After transpose we expect (N, C) per-row vectors where:
           * C == 5 + num_classes  => [x,y,w,h,obj_conf, cls_scores...]
           * C == 4 + num_classes  => [x,y,w,h, cls_scores...] (no obj_conf)
           * C == 6 and so on  => [x1,y1,x2,y2, conf, class_id]
        """
        if preds is None:
            return []

        # Normalize to numpy array
        preds = np.asarray(preds)

        if self.debug:
            print("[ONNXDetector] raw preds shape before reshape:", preds.shape)

        # If shape is (1, C, N) and C is small while N is large -> transpose to (1, N, C)
        if preds.ndim == 3 and preds.shape[0] == 1 and preds.shape[1] < preds.shape[2]:
            # common case from Ultralytics: (1, 84, 8400) => transpose to (1, 8400, 84)
            preds = preds.transpose(0, 2, 1)
            if self.debug:
                print("[ONNXDetector] transposed preds to:", preds.shape)

        # If preds is (1, N, C) -> squeeze batch dim
        if preds.ndim == 3 and preds.shape[0] == 1:
            preds = preds[0]

        # ensure 2D
        if preds.ndim == 1:
            preds = preds.reshape(1, -1)

        N, C = preds.shape
        if self.debug:
            print(f"[ONNXDetector] interpreting preds as (N,C)=({N},{C})")

        results = []

        # heuristic: number of classes (default COCO=80)
        num_classes_guess = max(80, C - 5)  # fallback

        # Case: format [x_center, y_center, w, h, obj_conf, cls_scores...]
        if C >= 6 and (C - 5) <= 100:  # reasonable num classes
            # if C == 5 + num_classes -> we expect obj_conf at index 4
            # if C == 4 + num_classes -> likely no obj_conf, class scores start at 4
            if C == 5 + 80 or C == 5 + (len(self.class_names)):
                has_obj = True
                cls_start = 5
            elif C == 4 + 80 or C == 4 + (len(self.class_names)):
                has_obj = False
                cls_start = 4
            else:
                # guess based on values: check column 4 statistics
                col4 = preds[:, 4]
                # if col4 values mostly > 1e-3 and <=1.0 -> probably obj_conf. If col4 has large values up to input_size -> maybe coordinate
                if np.nanmax(col4) <= 1.01:
                    has_obj = True
                    cls_start = 5
                else:
                    # assume no obj conf
                    has_obj = False
                    cls_start = 4

            # detect whether coordinates are normalized (<=1) or in letterbox pixel scale
            coord_max = preds[:, :4].max()
            normalized_coords = coord_max <= 1.01

            for i in range(N):
                row = preds[i]
                if has_obj:
                    obj_conf = float(row[4])
                    class_scores = row[5:]
                    class_id = int(np.argmax(class_scores))
                    class_conf = float(class_scores[class_id])
                    score = obj_conf * class_conf
                else:
                    class_scores = row[4:]
                    class_id = int(np.argmax(class_scores))
                    class_conf = float(class_scores[class_id])
                    score = class_conf

                if score < self.conf_thres:
                    continue

                x_c, y_c, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                if normalized_coords:
                    x_c *= self.input_size
                    y_c *= self.input_size
                    w   *= self.input_size
                    h   *= self.input_size

                x1 = x_c - w/2
                y1 = y_c - h/2
                x2 = x_c + w/2
                y2 = y_c + h/2

                # unpad & scale to original image
                x1 = (x1 - pad_x) / ratio
                x2 = (x2 - pad_x) / ratio
                y1 = (y1 - pad_y) / ratio
                y2 = (y2 - pad_y) / ratio

                results.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'conf': float(score),
                    'class_id': int(class_id),
                    'class_name': self.class_names.get(int(class_id), str(int(class_id)))
                })

            # apply classwise NMS and return
            return self._nms_and_format(results)
# ####################################################################################################################################
#         if C == 5:
#             # This format assumes a single class, so class_id is always 0.
#             for i in range(N):
#                 row = preds[i]
#                 conf = float(row[4])
                
#                 if conf < self.conf_thres:
#                     continue

#                 class_id = 0 # Hardcoded for single-class models
                
#                 x_c, y_c, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                
#                 x1 = x_c - w/2
#                 y1 = y_c - h/2
#                 x2 = x_c + w/2
#                 y2 = y_c + h/2

#                 # Unpad and scale to original image dimensions
#                 x1 = (x1 - pad_x) / ratio
#                 x2 = (x2 - pad_x) / ratio
#                 y1 = (y1 - pad_y) / ratio
#                 y2 = (y2 - pad_y) / ratio

#                 results.append({
#                     'bbox': [float(x1), float(y1), float(x2), float(y2)],
#                     'conf': conf,
#                     'class_id': class_id,
#                     'class_name': self.class_names.get(class_id, str(class_id))
#                 })
            
#             # Apply NMS and return the filtered detections
#             return self._nms_and_format(results)    
            #############################################################################################################

        # Case: [x1,y1,x2,y2, conf, class_id]
        if C == 6:
            coord_max = preds[:, :4].max()
            normalized_coords = coord_max <= 1.01
            for i in range(N):
                row = preds[i]
                x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                conf = float(row[4])
                class_id = int(row[5])
                if normalized_coords:
                    x1 *= self.input_size; x2 *= self.input_size; y1 *= self.input_size; y2 *= self.input_size
                x1 = (x1 - pad_x) / ratio; x2 = (x2 - pad_x) / ratio
                y1 = (y1 - pad_y) / ratio; y2 = (y2 - pad_y) / ratio
                if conf >= self.conf_thres:
                    results.append({'bbox':[x1,y1,x2,y2], 'conf':conf, 'class_id':class_id, 'class_name': self.class_names.get(class_id, str(class_id))})
            return self._nms_and_format(results)

        # Fallback: no parse -> return empty (debug prints already show shapes)
        if self.debug:
            print("[ONNXDetector] fallback: could not parse preds format. Returning empty.")
        return []


    def _nms_and_format(self, results):
        # results: list of dicts with bbox/conf/class_id
        if len(results) == 0:
            return []
        boxes = np.array([r['bbox'] for r in results])
        scores = np.array([r['conf'] for r in results])
        class_ids = np.array([r['class_id'] for r in results])
        out = []
        for cls in np.unique(class_ids):
            mask = class_ids == cls
            b = boxes[mask]
            s = scores[mask]
            idxs = nms_simple(b, s, self.iou_thres)
            for k in idxs:
                orig_idx = np.where(mask)[0][k]
                rr = results[orig_idx]
                out.append({'bbox':[float(x) for x in rr['bbox']], 'conf':float(rr['conf']), 'class_id': int(rr['class_id']), 'class_name': self.class_names.get(int(rr['class_id']), str(int(rr['class_id'])))})
        return out

    def detect(self, frame):
        # Fixed unpacking: preprocess returns (img, ratio, pad_x, pad_y)
        img, ratio, pad_x, pad_y = self.preprocess(frame)
        ort_inputs = {self.input_name: img}
        outputs = self.session.run(None, ort_inputs)
        preds = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        # interpret and convert
        dets = self._interpret_and_convert(preds, ratio=ratio, pad_x=pad_x, pad_y=pad_y, orig_shape=frame.shape[:2])
        if self.debug:
            print(f"[ONNXDetector] parsed {len(dets)} detections")
            # for i, d in enumerate(dets[:5]):
            #     print("  ->", i, d)
        return dets

