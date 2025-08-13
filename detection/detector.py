# detector.py
import numpy as np
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("ultralytics YOLO not available. Install 'ultralytics' or adapt to ONNXRuntime.")

class Detector:
    """
    Simple detector wrapper. By default tries to use ultralytics YOLOv8.
    You can replace this with an ONNXRuntime wrapper for deployment.
    """
    def __init__(self, model_path='yolov8n.pt', device='cpu', conf_thresh=0.15): # conf_thresh tunable, lower less aggressive
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO not available. Install 'ultralytics' or change detector implementation.")
        self.model = YOLO(model_path)   # use yolov8n.pt for speed; change to .onnx if needed with ONNX wrapper
        self.device = device
        self.conf_thresh = conf_thresh

        # mapping COCO ids -> names is included in the model prediction results
        # if you fine-tune your model with only the classes you need then adjust accordingly

    def detect(self, frame):
        """
        Detect objects in frame. Returns list of detections:
        [ { 'bbox': [x1,y1,x2,y2], 'conf': float, 'class_name': str } , ... ]
        """
        # ultralytics returns a list of Results; we pass frame (ndarray)
        results = self.model(frame, device=self.device, conf=self.conf_thresh, imgsz=640)
        dets = []
        # support single image inference
        if len(results) == 0:
            return dets

        r = results[0]
        # r.boxes has .xyxy numpy or torch
        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else np.array([])
        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.array([])
        cls_ids = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, 'cls') else np.array([])

        # get class names from model
        names = self.model.names

        for box, conf, cid in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = map(float, box)
            dets.append({'bbox': [x1, y1, x2, y2], 'conf': float(conf), 'class_name': names.get(cid, str(cid))})
        return dets
