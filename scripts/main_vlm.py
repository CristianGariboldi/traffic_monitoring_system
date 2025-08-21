#!/usr/bin/env python3

import argparse
import time
import cv2
import numpy as np
import json
import os
import threading
import traceback
from pathlib import Path
from PIL import Image, ImageOps

# detector / tracker / speed estimator imports - assume these exist in the repo
from detector_onnx import ONNXDetector as Detector
from tracker import CentroidTracker
from speed_estimator import SpeedEstimator, compute_homography_from_pairs

# filters (optional)
try:
    from filters import load_filter_from_config
    _HAS_FILTERS = True
except Exception:
    load_filter_from_config = None
    _HAS_FILTERS = False

# legacy white heuristic fallback
try:
    from color_filter import is_white_car as legacy_is_white_car
except Exception:
    legacy_is_white_car = None

ALLOWED_CLASSES = {'car', 'truck', 'bus', 'van', 'motorbike', 'motorcycle'}

def draw_box(frame, bbox, label=None, color=(0,255,0), thickness=2):
    x1,y1,x2,y2 = map(int, bbox)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, max(0,y1-18)), (x1+tw+6, y1), color, -1)
        cv2.putText(frame, label, (x1+2, max(0,y1-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)

def parse_args():
    p = argparse.ArgumentParser(description="Two-gate counting + optional VLM scene analysis")
    p.add_argument('--source', '-s', default="./data/Video.mp4", help='video file or camera index')
    p.add_argument('--model', default='./models/yolo11n.onnx', help='ONNX detection model path')
    p.add_argument('--conf', type=float, default=0.35, help='detection confidence threshold (0..1)')
    p.add_argument('--debug', action='store_true', help='print debug info')
    p.add_argument('--filter-config', default="./config/filters.yaml", help='path to filters YAML/JSON config')
    p.add_argument('--filter-name', default=None, help='named filter in config (if omitted, no filter => count all)')
    p.add_argument('--white-only', action='store_true', help='legacy: if set, count only white cars using old heuristic')
    p.add_argument('--auto-calib', action='store_true', help='auto-calibrate gate positions from first frames')
    p.add_argument('--calib-frames', type=int, default=60, help='frames for calibration if --auto-calib')
    p.add_argument('--calib-top-q', type=float, default=0.65, help='top quantile for gate placement (0..1)')
    p.add_argument('--calib-bottom-q', type=float, default=0.80, help='bottom quantile for gate placement (0..1)')
    p.add_argument('--seq-window', type=float, default=3.0, help='seconds allowed between two gate entries')
    p.add_argument('--fps-sync', action='store_true', help='match display to source FPS')
    p.add_argument('--homography', default="./config/homography.json",
                help='path to homography JSON file with image_points and world_points (meters)')
    p.add_argument('--speed-smooth-alpha', type=float, default=0.6,
                help='EMA alpha for speed smoothing (0..1). If 0 use median smoothing.')

    # VLM options
    p.add_argument('--vlm-enable', action='store_true', help='enable periodic VLM-based scene analysis')
    p.add_argument('--vlm-interval', type=float, default=10.0, help='seconds between VLM calls (default 8s)')
    p.add_argument('--vlm-model-id', default="HuggingFaceTB/SmolVLM-256M-Instruct",
                help='VLM model id (kept for processor/config loading; kept default as example)')
    p.add_argument('--vlm-onnx-dir', default="onnx", help='directory with onnx files (vision_encoder.onnx, embed_tokens.onnx, decoder_model_merged.onnx)')
    return p.parse_args()


# ----------------------
# VLM worker (loads sessions and runs generation loop periodically)
# ----------------------
class VLMWorker:
    """
    Background thread that periodically runs a VLM generation on the most recent frame.
    The structure of the VLM inference is intentionally similar to the example you provided.
    """

    def __init__(self, onnx_dir: str, model_id: str = None, interval: float = 8.0, debug: bool = False):
        self.onnx_dir = Path(onnx_dir)
        self.model_id = model_id
        self.interval = float(interval)
        self.debug = bool(debug)

        self._lock = threading.Lock()
        self._last_frame = None  # BGR numpy frame (cv2)
        self._last_result = "No scene analysis yet."
        self._last_ts = 0.0
        self._alert = False
        self._running = False
        self._thread = None

        # VLM runtime objects (populated in load)
        self.config = None
        self.processor = None
        self.vision_session = None
        self.embed_session = None
        self.decoder_session = None
        self.eos_token_id = None
        self.image_token_id = None
        self.num_key_value_heads = None
        self.head_dim = None
        self.num_hidden_layers = None

        # keywords that trigger alert (lowercased)
        self.alert_keywords = [
            "accident", "collision", "stalled", "stopping", "stuck", "fire", "smoke",
            "flood", "pothole", "obstacle", "pedestrian", "cyclist", "bicycle", "roadblock",
            "blocked", "breakdown", "congestion", "traffic jam", "jammed", "crash", "collision"
        ]

    def load(self):
        """Load VLM config/processor and ONNX sessions; keep structure like provided example."""
        try:
            # lazy import to avoid importing heavy libs when VLM is disabled
            from transformers import AutoConfig, AutoProcessor
            from transformers.image_utils import load_image  # kept for structural similarity
            import onnxruntime
            import numpy as _np
        except Exception as e:
            raise RuntimeError("Failed to import VLM dependencies: " + str(e))

        # --- 1. Load Models and Configuration ---
        if self.debug:
            print("[VLM] Loading models and processor...")

        # Note: the example used a Hugging Face model id to load config/processor;
        # we keep the same shape here so your existing processor files / config work.
        self.config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

        # ONNX sessions
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        vis_p = str(self.onnx_dir / "vision_encoder.onnx")
        emb_p = str(self.onnx_dir / "embed_tokens.onnx")
        dec_p = str(self.onnx_dir / "decoder_model_merged.onnx")
        if self.debug:
            print(f"[VLM] Loading ONNX: {vis_p}, {emb_p}, {dec_p}")

        self.vision_session = onnxruntime.InferenceSession(vis_p, providers=providers)
        self.embed_session = onnxruntime.InferenceSession(emb_p, providers=providers)
        self.decoder_session = onnxruntime.InferenceSession(dec_p, providers=providers)

        # extract a few config values used in generation loop
        num_key_value_heads = self.config.text_config.num_key_value_heads
        head_dim = self.config.text_config.head_dim
        num_hidden_layers = self.config.text_config.num_hidden_layers
        eos_token_id = self.config.text_config.eos_token_id
        image_token_id = self.config.image_token_id

        # store them
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id

        if self.debug:
            print("[VLM] Models loaded successfully.")

    def start(self):
        if self._running:
            return
        try:
            self.load()
        except Exception as e:
            print("[VLM] Failed to load VLM:", e)
            if self.debug:
                traceback.print_exc()
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if self.debug:
            print("[VLM] Worker thread started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)
            self._thread = None

    def submit_frame(self, frame_bgr):
        """Main loop can call this every frame; we just keep the latest frame."""
        with self._lock:
            # copy minimal (avoid referencing external array that will be overwritten)
            self._last_frame = frame_bgr.copy()
            # mark last_ts to prefer fresh frames
            self._last_ts = time.time()

    def get_latest_result(self):
        with self._lock:
            return self._last_result, self._last_ts, self._alert

    # ---------- internal helpers ----------
    def _text_prompt_for_frame(self):
        """Return the messages/prompt to send to the VLM (you can tweak wording here)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text":
                        "Describe in detail the traffic and the environment in this image."}
                        # "You are a traffic monitoring system and I am showing you a camera image showing a highway with some vehicles. Describe the traffic situation in this image, focusing on potential dangerous actions. "
                        # "Focus on detecting accidents or safety-critical issues."
                        # "If you see cars crashing to each other or motorcycles falling down, describe it clearly"
                        # "Tell me if the traffic showed in this image is busy or not, and the environment conditions (for example raining, cloudy, snow, sunny) "}
                        # "(accident, stalled vehicle, pedestrian on road, heavy rain/flooding, smoke, fire, or other hazards)? "
                        # "If you detect a safety-critical issue, include the word 'ALERT'."}
                ]
            },
        ]
        return messages

    def _run_vlm_on_pil(self, pil_image):
        """
        Run the VLM generation loop on a PIL image and return a cleaned string.
        Keeps the structure of your example but:
         - ensures types are numpy-compatible,
         - cleans common VLM tokens from the output,
         - gracefully handles empty outputs.
        """
        # local imports to avoid heavy imports when VLM disabled
        from transformers import AutoProcessor
        import numpy as _np

        messages = self._text_prompt_for_frame()
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[pil_image], return_tensors="np")

        batch_size = int(inputs['input_ids'].shape[0])

        # initialize KV cache exactly like example
        past_key_values = {
            f'past_key_values.{layer}.{kv}': _np.zeros([batch_size, self.num_key_value_heads, 0, self.head_dim], dtype=_np.float32)
            for layer in range(self.num_hidden_layers)
            for kv in ('key', 'value')
        }

        image_features = None
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = _np.cumsum(attention_mask, axis=-1) - 1

        max_new_tokens = 1280
        # Keep same initial shape as the example: (batch, 0)
        generated_tokens = _np.empty((batch_size, 0), dtype=_np.int64)

        for i in range(max_new_tokens):
            # text embeddings
            inputs_embeds = self.embed_session.run(None, {'input_ids': input_ids})[0]

            if image_features is None:
                # keep original boolean conversion
                image_features = self.vision_session.run(
                    None,
                    {
                        'pixel_values': inputs['pixel_values'],
                        'pixel_attention_mask': inputs['pixel_attention_mask'].astype(_np.bool_)
                    }
                )[0]
                # replace image token embedding with image features
                inputs_embeds[inputs['input_ids'] == self.image_token_id] = image_features.reshape(-1, image_features.shape[-1])

            # run decoder
            logits_and_present = self.decoder_session.run(None, dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **past_key_values,
            ))
            logits = logits_and_present[0]
            present_key_values = logits_and_present[1:]

            # next token (argmax)
            next_token_id = logits[:, -1:].argmax(-1, keepdims=False)  # shape (batch,)
            # prepend/append as column
            generated_tokens = _np.concatenate([generated_tokens, next_token_id.reshape(batch_size, 1).astype(_np.int64)], axis=1)

            # streaming debug (optional)
            if self.debug:
                try:
                    token_text = self.processor.decode(next_token_id[0])
                    print("[VLM stream token]", token_text, end='', flush=True)
                except Exception:
                    pass

            # stop if eos generated for all batch elements
            if (next_token_id == self.eos_token_id).all():
                break

            # prepare next iteration
            input_ids = next_token_id.reshape(batch_size, 1)
            attention_mask = _np.concatenate([attention_mask, _np.ones_like(input_ids)], axis=-1)
            # position_ids update consistent with example (small hack to keep shape)
            position_ids = _np.array([[position_ids[:, -1][0] + 1]])

            # update KV cache in insertion order (should match present_key_values)
            for j, key in enumerate(list(past_key_values.keys())):
                past_key_values[key] = present_key_values[j]

        # final decode (safe)
        try:
            final_answer = self.processor.batch_decode(generated_tokens)[0]
        except Exception:
            # fallback: empty string if decoder fails
            final_answer = ""

        # cleanup common model-control tokens and whitespace
        if final_answer is None:
            final_answer = ""
        cleaned = final_answer.replace('<end_of_utterance>', '').replace('<|im_end|>', '').strip()
        # remove repeated trailing tokens sometimes emitted like " <|im_end|><|im_end|>"
        while cleaned.endswith('<|im_end|>') or cleaned.endswith('<end_of_utterance>'):
            cleaned = cleaned.rsplit('<', 1)[0].strip()

        return cleaned

    def _detect_alert_keywords(self, txt: str) -> bool:
        s = (txt or "").lower()
        for kw in self.alert_keywords:
            if kw in s:
                return True
        return False

    def _loop(self):
        """Background loop: take latest frame, run VLM, store result, and PRINT to terminal."""
        while self._running:
            start = time.time()
            frame_bgr = None
            with self._lock:
                if self._last_frame is not None:
                    frame_bgr = self._last_frame.copy()

            if frame_bgr is not None:
                try:
                    # convert to PIL Image RGB
                    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                    # run vlm
                    answer = self._run_vlm_on_pil(pil)

                    # detect keywords
                    alert = self._detect_alert_keywords(answer)

                    with self._lock:
                        # store cleaned answer and alert flag
                        self._last_result = answer.strip() if answer is not None else ""
                        self._alert = bool(alert)
                        self._last_ts = time.time()

                    # ALWAYS print result to terminal for user visibility
                    ts_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self._last_ts))
                    print(f"\n[VLM @ {ts_text}] {self._last_result}\n")
                    if self.debug:
                        # also print raw first 400 chars for debugging
                        print("[VLM debug] raw (first 400 chars):", (self._last_result or "")[:400])

                except Exception as e:
                    with self._lock:
                        self._last_result = f"VLM inference failed: {e}"
                        self._alert = False
                        self._last_ts = time.time()
                    print("[VLM] inference error:", e)
                    if self.debug:
                        traceback.print_exc()

            elapsed = time.time() - start
            to_sleep = max(0.1, self.interval - elapsed)
            time.sleep(to_sleep)


# ----------------------
# utility drawing for VLM overlay
# ----------------------
def draw_scene_overlay(frame, text, alert=False):
    """
    Draw a compact Scene Analysis overlay at top-left.
    - text: string (possibly multiple sentences)
    - alert: if True, draw red header
    """
    H, W = frame.shape[:2]
    # prepare small multiline text box
    lines = []
    # naive wrap: split on sentences / commas, then cap length
    for seg in text.replace('\n', ' ').split('. '):
        seg = seg.strip()
        if not seg:
            continue
        # chunk long sentences
        while len(seg) > 50:
            lines.append(seg[:50].strip())
            seg = seg[50:].strip()
        if seg:
            lines.append(seg)
        if len(lines) >= 6:
            break
    if len(lines) == 0:
        lines = ["No analysis available."]

    # box dims
    pad = 8
    line_h = 18
    box_h = pad*2 + line_h * len(lines) + 20
    box_w = min(W - 20, 640)
    # header color
    header_color = (0, 165, 0) if not alert else (0, 0, 255)
    # box background
    cv2.rectangle(frame, (6, 6), (6 + box_w, 6 + box_h), (30, 30, 30), -1)
    # header strip
    cv2.rectangle(frame, (6, 6), (6+box_w, 6+28), header_color, -1)
    cv2.putText(frame, "Scene Analysis", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    # text lines
    y = 6 + 28 + 6
    for ln in lines:
        cv2.putText(frame, ln, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)
        y += line_h


# ----------------------
# main script (same high-level flow as your main_2)
# ----------------------
def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("Cannot open source:", args.source)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_time = 1.0 / fps

    # create detector (ONNXDetector)
    detector = Detector(args.model, input_size=640, providers=['CPUExecutionProvider'], conf_thres=args.conf, debug=args.debug)
    tracker = CentroidTracker(max_missed=12, max_distance=140)

    # load filter if requested
    filter_obj = None
    if args.filter_config and args.filter_name:
        if not _HAS_FILTERS:
            print("Warning: filters module not available. Ignoring --filter-config.")
        else:
            try:
                filter_obj = load_filter_from_config(args.filter_config, args.filter_name)
                print(f"Loaded filter '{args.filter_name}' from {args.filter_config}")
            except Exception as e:
                print("Failed to load filter config:", e)
                filter_obj = None

    legacy_white_mode = bool(args.white_only) and (filter_obj is None)

    count_in = 0
    count_out = 0
    seq_window = float(args.seq_window)

    # calibration buffers & settings
    frame_idx = 0
    ys = []
    auto_calib = bool(args.auto_calib)
    calib_frames = max(1, int(args.calib_frames))
    top_q = float(args.calib_top_q)
    bottom_q = float(args.calib_bottom_q)
    default_top_frac = 0.65
    default_bottom_frac = 0.70
    top_y = None
    bottom_y = None

    prev_time = time.time()

    # Setup speed estimator (homography or fallback)
    H_homography = None
    if args.homography:
        try:
            with open(args.homography, 'r') as fh:
                j = json.load(fh)
            img_pts = j['image_points']
            world_pts = j['world_points']
            H_homography = compute_homography_from_pairs(img_pts, world_pts)
            print("Loaded homography from", args.homography)
        except Exception as e:
            print("Failed to load homography:", e)
            H_homography = None

    fallback_m_per_px = None
    speed_est = SpeedEstimator(H=H_homography, fallback_m_per_px=fallback_m_per_px, history_len=8,
                            smooth_alpha=(args.speed_smooth_alpha if args.speed_smooth_alpha>0 else None))

    # VLM worker: start only if requested
    vlm_worker = None
    if args.vlm_enable:
        vlm_worker = VLMWorker(onnx_dir=args.vlm_onnx_dir, model_id=args.vlm_model_id, interval=args.vlm_interval, debug=args.debug)
        vlm_worker.start()
        print("[Main] VLM enabled. Interval (s):", args.vlm_interval)

    try:
        while True:
            start_frame = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # KEEP a pristine copy of the raw frame for the VLM (do this BEFORE any drawing)
            raw_frame_for_vlm = frame  # no copy here; submit_frame makes its own copy
            if vlm_worker is not None:
                # optional: downscale while preserving aspect ratio to reduce VLM IO
                try:
                    h, w = raw_frame_for_vlm.shape[:2]
                    target_w = 640
                    scale = min(1.0, target_w / float(w))
                    if scale < 1.0:
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        vlm_img = cv2.resize(raw_frame_for_vlm, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    else:
                        vlm_img = raw_frame_for_vlm
                except Exception:
                    vlm_img = raw_frame_for_vlm
                # submit non-blocking; worker will copy the array under lock
                try:
                    vlm_worker.submit_frame(vlm_img)
                except Exception:
                    if args.debug:
                        print("[Main] Warning: failed to submit frame to VLM worker")


            H_img, W_img = frame.shape[:2]

            # ----------------------------
            # Auto-calibration phase
            # ----------------------------
            if auto_calib and frame_idx < calib_frames:
                dets_tmp = detector.detect(frame)
                for d in dets_tmp:
                    cls = str(d['class_name']).lower()
                    if cls not in ALLOWED_CLASSES:
                        continue
                    x1,y1,x2,y2 = d['bbox']
                    cy = int(y2)
                    ys.append(cy)
                frame_idx += 1
                cv2.putText(frame, f'Auto-calibrating gates: {frame_idx}/{calib_frames}', (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.line(frame, (0, int(H_img*default_top_frac)), (W_img-1, int(H_img*default_top_frac)), (200,200,0), 1)
                cv2.line(frame, (0, int(H_img*default_bottom_frac)), (W_img-1, int(H_img*default_bottom_frac)), (200,200,0), 1)
                cv2.imshow('output', frame)
                if args.fps_sync:
                    elapsed = time.time() - start_frame
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                continue

            if auto_calib and top_y is None and frame_idx >= calib_frames:
                if len(ys) >= 4:
                    top_y = int(np.quantile(ys, top_q))
                    bottom_y = int(np.quantile(ys, bottom_q))
                    if top_y >= bottom_y:
                        med = int(np.median(ys))
                        span = max(10, int(0.12 * H_img))
                        top_y = max(8, med - span)
                        bottom_y = min(H_img-8, med + span)
                else:
                    top_y = int(H_img * default_top_frac)
                    bottom_y = int(H_img * default_bottom_frac)
                auto_calib = False
                print(f"Auto-calibrated gates: top_y={top_y}, bottom_y={bottom_y}  (collected {len(ys)} samples)")

            if top_y is None:
                top_y = int(H_img * default_top_frac)
            if bottom_y is None:
                bottom_y = int(H_img * default_bottom_frac)

            gate_half_h = max(8, int(H_img * 0.02))
            top_gate = (0, max(0, top_y - gate_half_h), W_img - 1, min(H_img - 1, top_y + gate_half_h))
            bottom_gate = (0, max(0, bottom_y - gate_half_h), W_img - 1, min(H_img - 1, bottom_y + gate_half_h))

            # ----------------------------
            # Detection & tracking
            # ----------------------------
            dets = detector.detect(frame)
            all_dets_for_tracker = []
            for d in dets:
                cls = str(d['class_name']).lower()
                if cls not in ALLOWED_CLASSES:
                    continue
                is_countable = True
                if filter_obj is not None:
                    try:
                        is_countable = bool(filter_obj.match(frame, d['bbox'], cls))
                    except Exception:
                        is_countable = False
                elif legacy_is_white_car and args.white_only and cls == 'car':
                    is_countable = bool(legacy_is_white_car(frame, d['bbox']))
                else:
                    is_countable = True
                all_dets_for_tracker.append({'bbox': d['bbox'], 'class_name': cls, 'is_countable': is_countable})
                if args.debug and not is_countable:
                    print(f"DEBUG skip countable: cls={cls}, bbox={d['bbox']}, is_countable={is_countable}")

            tracks = tracker.update(all_dets_for_tracker)

            # draw gates
            cv2.rectangle(frame, (top_gate[0], top_gate[1]), (top_gate[2], top_gate[3]), (0,255,0), 2)
            cv2.rectangle(frame, (bottom_gate[0], bottom_gate[1]), (bottom_gate[2], bottom_gate[3]), (0,0,255), 2)

            # timestamp in seconds
            video_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = (video_msec / 1000.0) if video_msec and video_msec > 0 else time.time()

            # fallback m_per_px if no homography (set from first big car)
            if speed_est.H is None and speed_est.fallback_m_per_px is None and hasattr(args, 'ref_car_length') and getattr(args, 'ref_car_length', None):
                for det in all_dets_for_tracker:
                    if det['class_name'] == 'car':
                        x1,y1,x2,y2 = det['bbox']
                        pix_w = max(1, int(x2) - int(x1))
                        speed_est.fallback_m_per_px = float(args.ref_car_length) / float(pix_w)
                        if args.debug:
                            print("Set fallback m_per_px =", speed_est.fallback_m_per_px, "from pix width", pix_w)
                        break

            # update per-track info
            for t in tracks:
                # color (kept same as before)
                if args.debug:
                    print(f"TRACK {t.id}: centroid={t.centroid} hits={t.hits} miss={t.miss} is_countable={t.is_countable} counted={t.counted}")

                inside_top_now = (t.centroid[1] >= top_gate[1] and t.centroid[1] <= top_gate[3])
                was_inside_top = ('top' in t.inside_gates)
                if inside_top_now and not was_inside_top:
                    t.enter_gate('top', ts)
                elif not inside_top_now and was_inside_top:
                    t.exit_gate('top')

                inside_bottom_now = (t.centroid[1] >= bottom_gate[1] and t.centroid[1] <= bottom_gate[3])
                was_inside_bottom = ('bottom' in t.inside_gates)
                if inside_bottom_now and not was_inside_bottom:
                    t.enter_gate('bottom', ts)
                elif not inside_bottom_now and was_inside_bottom:
                    t.exit_gate('bottom')

                # speed estimation
                x1,y1,x2,y2 = t.bbox
                img_pt = (((float(x1)+float(x2))/2.0), float(y2))
                speed_kmh = speed_est.add_observation(t.id, img_pt, ts=ts)
                t.speed_kmh = speed_kmh if speed_kmh is not None else None

                # counting logic (unchanged)
                if not t.counted:
                    recent = [ (g, tt) for g, tt in t.gate_history if (ts - tt) <= seq_window ]
                    if args.debug and recent:
                        print(f"  Track {t.id} recent gates: {recent}")
                    if len(recent) >= 2:
                        prev_gate, prev_ts = recent[-2]
                        last_gate, last_ts = recent[-1]
                        if prev_gate == 'top' and last_gate == 'bottom':
                            if filter_obj is not None:
                                if t.is_countable:
                                    count_in += 1
                                    t.counted = True
                            elif legacy_is_white_car and args.white_only:
                                if t.is_countable:
                                    count_in += 1
                                    t.counted = True
                            else:
                                count_in += 1
                                t.counted = True
                        elif prev_gate == 'bottom' and last_gate == 'top':
                            if filter_obj is not None:
                                if t.is_countable:
                                    count_out += 1
                                    t.counted = True
                            elif legacy_is_white_car and args.white_only:
                                if t.is_countable:
                                    count_out += 1
                                    t.counted = True
                            else:
                                count_out += 1
                                t.counted = True

            # cleanup speed history for absent tracks
            active_ids = set([t.id for t in tracks])
            for tid in list(speed_est.hist.keys()):
                if tid not in active_ids:
                    speed_est.remove_track(tid)

            # find fastest
            fastest = None
            fastest_v = -1.0
            for t in tracks:
                if getattr(t, 'speed_kmh', None) is not None:
                    if t.speed_kmh > fastest_v:
                        fastest_v = t.speed_kmh
                        fastest = t

            # draw boxes and labels
            for t in tracks:
                if t.class_name == 'car':
                    base_color = (0, 255, 0) if t.is_countable else (0, 255, 255)
                elif t.class_name in ('truck','bus','van'):
                    base_color = (0, 165, 255)
                elif t.class_name in ('motorbike','motorcycle'):
                    base_color = (255, 0, 255)
                else:
                    base_color = (200,200,200)

                if fastest is not None and t.id == fastest.id:
                    box_color = (0,0,255)
                    thickness = 3
                else:
                    box_color = base_color
                    thickness = 2

                speed_txt = f"{t.speed_kmh:.1f}km/h" if (getattr(t, 'speed_kmh', None) is not None) else ""
                label = f'ID:{t.id} {t.class_name or ""} {speed_txt}'
                draw_box(frame, t.bbox, label=label, color=box_color, thickness=thickness)
                try:
                    cv2.circle(frame, t.centroid, 4, box_color, -1)
                except:
                    pass

            # overlay counters & fps
            mode_desc = args.filter_name if args.filter_name else ('white-only' if legacy_is_white_car and args.white_only else 'all')
            cv2.putText(frame, f'Incoming ({mode_desc}): {count_in}', (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, f'Outgoing ({mode_desc}): {count_out}', (12,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            now = time.time()
            fps_val = 1.0 / (now - prev_time + 1e-8)
            prev_time = now
            cv2.putText(frame, f'FPS: {fps_val:.1f}', (12,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

            # submit the frame to VLM worker (non-blocking - it only stores the latest frame)
            if vlm_worker is not None:
                # vlm_worker.submit_frame(frame)

                # get latest VLM result (thread-safe)
                vlm_text, vlm_ts, vlm_alert = vlm_worker.get_latest_result()
                # if result is recent (within 2*interval) show else show placeholder
                if (time.time() - vlm_ts) < max(2.0, args.vlm_interval * 2.0):
                    draw_scene_overlay(frame, vlm_text, alert=vlm_alert)
                else:
                    draw_scene_overlay(frame, "Scene Analysis: pending...", alert=False)

            # show and sync fps
            cv2.imshow('output', frame)
            if args.fps_sync:
                elapsed = time.time() - start_frame
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('p'):
                cv2.waitKey(0)

    finally:
        if vlm_worker is not None:
            vlm_worker.stop()
            time.sleep(0.2)
        cap.release()
        cv2.destroyAllWindows()
        print("Finished. incoming:", count_in, "outgoing:", count_out)


if __name__ == '__main__':
    main()
