# upload -> predict -> inference
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from PIL import Image
import io, os, itertools, base64, time

import torch
import imageio.v2 as imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt

from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import onnxruntime as ort
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

def image_to_base64_bmp(binary_mask):
    pil = Image.fromarray(binary_mask)
    buf = io.BytesIO()
    pil.save(buf, format="BMP")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


app = FastAPI()

# ---------------- Config ----------------
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_SIZE_MB = 30

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "./weights/best.pt")
SAM_CKPT     = os.environ.get("SAM_CKPT", "./weights/sam_vit_h_4b8939.pth")
SAM_ONNX     = os.environ.get("SAM_ONNX", "./weights/sam_onnx_example.onnx")
SAM_TYPE     = os.environ.get("SAM_TYPE", "vit_h")  # vit_h | vit_l | vit_b
YOLO_CONF    = float(os.environ.get("YOLO_CONF", "0.5"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# ---------------- Globals ----------------
IMAGES = {}  # id -> {"bytes":..., "content_type":..., "filename":...}
_id_counter = itertools.count(1)

# YOLO
YOLO_MODEL = YOLO(YOLO_WEIGHTS).to(DEVICE)

# SAM (torch for embeddings)
_sam_torch = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(DEVICE)
_sam_torch.eval()
PREDICTOR = SamPredictor(_sam_torch)

# SAM (onnx decoder)
def _make_onnx_session(model_path: str):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
    return ort.InferenceSession(model_path, providers=providers)

ORT_SESSION = _make_onnx_session(SAM_ONNX)

@app.post("/image")
async def upload_image(file: UploadFile = File(...)):
    start = time.perf_counter()
    global latest_image_bytes, latest_yolo_results, latest_masks_b64, latest_combined_b64
    latest_image_bytes = await file.read()

    pil = Image.open(io.BytesIO(latest_image_bytes)).convert("RGB")
    results = YOLO_MODEL.predict(pil, conf=YOLO_CONF, verbose=False)[0]
    app.state.yolo_raw_results = results

    # YOLO 추론 결과 저장
    if not results.boxes:
        latest_yolo_results = {"message": "no objects detected"}
        latest_masks_b64 = []
        latest_combined_b64 = ""
        return {"message": "Image uploaded, but no objects detected."}

    # YOLO 결과 변환
    xyxy = results.boxes.xyxy.cpu().numpy().tolist()
    xywh = results.boxes.xywh.cpu().numpy().tolist()
    clss  = results.boxes.cls.cpu().numpy().astype(int).tolist()
    confs = results.boxes.conf.cpu().numpy().tolist()
    names = results.names

    detections = []
    bbox_centers = []
    for i in range(len(clss)):
        detections.append({
            "bbox_xyxy": xyxy[i],
            "centers": xywh[i],
            "class_id": clss[i],
            "class_name": names[clss[i]],
            "confidence": confs[i]
        })
        bbox_centers.append([xywh[i][0], xywh[i][1]])

    latest_yolo_results = {
        "num_detections": len(detections),
        "detections": detections
    }

    # SAM inference 시작
    image = np.array(pil)
    PREDICTOR.set_image(image)
    image_embedding = PREDICTOR.get_image_embedding().cpu().numpy()
    input_box = results.boxes.xyxy.cpu().numpy()
    input_point = np.array(bbox_centers)
    input_label = np.array([1]*len(bbox_centers))

    all_masks = []
    masks_b64 = []
    for i, box, center, label in zip(range(len(bbox_centers)), input_box, input_point, input_label):
        center = center.reshape(1,2)
        label = np.array([1])
        onnx_box_coords = box.reshape(2, 2)
        onnx_box_labels = np.array([2,3])
        onnx_coord = np.concatenate([center, onnx_box_coords], axis=0)[None, :, :]
        onnx_label = np.concatenate([label, onnx_box_labels], axis=0)[None, :].astype(np.float32)

        onnx_coord = PREDICTOR.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)

        ort_inputs = {
            "image_embeddings": image_embedding,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
            "has_mask_input": np.zeros(1, dtype=np.float32),
            "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
        }

        masks, _, _ = ORT_SESSION.run(None, ort_inputs)
        masks = masks > PREDICTOR.model.mask_threshold
        all_masks.append(masks[0][0])
        binary_mask = masks[0][0].astype(np.uint8) * 255
        masks_b64.append(image_to_base64_bmp(binary_mask))

    combined_mask = np.zeros_like(all_masks[0], dtype=np.uint8)
    for mask in all_masks:
        combined_mask = np.logical_or(combined_mask, mask)
    binary_combined = combined_mask.astype(np.uint8) * 255
    combined_b64 = image_to_base64_bmp(binary_combined)

    latest_masks_b64 = masks_b64
    latest_combined_b64 = combined_b64

    end = time.perf_counter()
    print(f"exec. time: {(end-start)*1000:.2f} ms")

    return {"message": "Image uploaded and processed successfully."}

@app.get("/yolo")
def get_yolo_result():
    if latest_yolo_results is None:
        raise HTTPException(status_code=404, detail="No image processed yet.")
    return latest_yolo_results

@app.get("/masks")
def get_mask_result():
    if latest_masks_b64 is None or latest_combined_b64 is None:
        raise HTTPException(status_code=404, detail="No image processed yet.")
    return {
        "message": "Mask results from previous inference",
        "num_instances": len(latest_masks_b64),
        "masks_bmp_b64": latest_masks_b64,
        "combined_mask_b64": latest_combined_b64
    }
