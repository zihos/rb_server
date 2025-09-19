# upload -> predict -> inference
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io, os, itertools, base64, time

import torch
import numpy as np
import cv2

from ultralytics import YOLO
import dice_utils
from segment_anything import sam_model_registry, SamPredictor



def image_to_base64_bmp(binary_mask):
    pil = Image.fromarray(binary_mask)
    buf = io.BytesIO()
    pil.save(buf, format="BMP")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ---------------- Config ----------------
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_SIZE_MB = 30

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "./weights/dice_best.engine")
# YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "/mnt/ssd/dice_weights/dice_yolov8.engine")
# FASTSAM_TRT  = os.environ.get("FASTAM_TRT", "/mnt/ssd/dice_weights/fastsam_dice.trt" )
FASTSAM_TRT  = os.environ.get("FASTAM_TRT", "/home/nvidia/FastSam_Awsome_TensorRT/dice_dark_best.trt" )
YOLO_CONF    = float(os.environ.get("YOLO_CONF", "0.5"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# ---------------- Globals ----------------
IMAGES = {}  # id -> {"bytes":..., "content_type":..., "filename":...}
_id_counter = itertools.count(1)

# YOLO
YOLO_MODEL = YOLO(YOLO_WEIGHTS)
FASTSAM_MODEL = dice_utils.FastSam(model_weights=FASTSAM_TRT)

SAM_CKPT     = os.environ.get("SAM_CKPT", "./weights/sam_vit_h_4b8939.pth")
SAM_ONNX     = os.environ.get("SAM_ONNX", "./weights/sam_onnx_example.onnx")
SAM_TYPE     = os.environ.get("SAM_TYPE", "vit_h")  # vit_h | vit_l | vit_b
_sam_torch = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(DEVICE)
_sam_torch.eval()

app = FastAPI()

@app.post("/image")
async def upload_image(file: UploadFile = File(...)):
    start = time.perf_counter()
    global latest_image_bytes, latest_yolo_results, latest_masks_b64, latest_combined_b64
    
    latest_image_bytes = await file.read()
    # pil = Image.open(io.BytesIO(latest_image_bytes)).convert("RGB")
    image_cv2 = np.frombuffer(latest_image_bytes, np.uint8)
    image_cv2 = cv2.imdecode(image_cv2, cv2.IMREAD_COLOR)
    print(image_cv2.shape)

    # gamma correction
    gamma=1.5
    inv_gamma = 1.0 / gamma
    table = np.array([(i/255.0)**inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    gamma_corrected_image = cv2.LUT(image_cv2, table)
    cv2.imwrite("./gamma_corrected_image.bmp", gamma_corrected_image)
    yolo_start=time.perf_counter()
    results = YOLO_MODEL.predict(gamma_corrected_image, conf=YOLO_CONF, verbose=False)[0]
    yolo_end=time.perf_counter()
    print(f"[YOLO] exec. time: {(yolo_end-yolo_start)*1000:.2f} ms, yolo results: {len(results)}")

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

    print(f"[YOLO] number of detected objects: {len(xyxy)}")

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

    # fastsam trt engine inference start
    # image_cv2 = np.frombuffer(latest_image_bytes, np.uint8)

    masks = FASTSAM_MODEL.segment(gamma_corrected_image, xyxy)

    if masks == None:
        print("[FASTSAM] no masks detected")
        latest_masks_b64 = []
        latest_combined_b64 = []

        return {"message": "No Masks !!!"}
    else:
        print("[FASTSAM] mask shape: ", masks.shape)

        masks_b64 = []
        combined_mask = np.zeros(masks.shape[1:], dtype=np.uint8)
        for i in range(masks.shape[0]):
            mask = masks[i]
            mask_np = mask.detach().cpu().numpy().astype(np.uint8)
            mask_bin = (mask_np > 0).astype(np.uint8) * 255
            masks_b64.append(image_to_base64_bmp(mask_bin))

            combined_mask = np.maximum(combined_mask, mask_bin)
        
        combined_b64 = image_to_base64_bmp(combined_mask)

        latest_masks_b64 = masks_b64
        latest_combined_b64 = combined_b64

        end = time.perf_counter()
        print(f"TOTAL exec. time: {(end-start)*1000:.2f} ms")

        cv2.imwrite("./combined_mask.bmp", combined_mask)

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
