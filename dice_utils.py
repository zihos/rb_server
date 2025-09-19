import time
import os
import cv2
import numpy as np
from exec_backends.trt_loader import TrtModelNMS
import torch
from utils import overlay, segment_everything
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import ops
from random import randint
import time
import ultralytics
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any

print("dice_utils.py", ultralytics.__version__)


device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

def save_binray_masks(masks: torch.Tensor, out_dir = "masks_out"):
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    combined_mask = np.zeros(masks.shape[1:], dtype=np.uint8)

    for i in range(masks.shape[0]):
        mask = masks[i]
        mask_np = mask.detach().cpu().numpy().astype(np.uint8)
        mask_bin = (mask_np > 0).astype(np.uint8) * 255

        out_path = os.path.join(out_dir, f"mask_{i}.png")
        cv2.imwrite(out_path, mask_bin)
        print(f"Saved: {out_path}")

        combined_mask = np.maximum(combined_mask, mask_bin)
    
    combined_path = os.path.join(out_dir, "mask_combined.png")
    cv2.imwrite(combined_path, combined_mask)
    print(f"Saved combined mask: {combined_path}")

def seleck_mask_by_bbox(
        masks:torch.Tensor, 
        bboxes: List[Tuple[float, float, float, float]],
        threshold: float = 0.5,
        topk: int=1,
        return_indices:bool=True
        ) -> List[Dict[str, Any]]:
    
    # assert masks.ndim == 3, 
    N, H, W = masks.shape

    masks_bin = (masks.detach().cpu()>0).to(torch.uint8)

    results = []

    for (x1, y1, x2, y2) in bboxes:
        x1c = max(0, min(int(round(x1)), W))
        y1c = max(0, min(int(round(y1)), H))
        x2c = max(0, min(int(round(x2)), W))
        y2c = max(0, min(int(round(y2)), H))

        if x2c <= x1c or y2c<=y1c:
            res = {'bbox': (x1, y1, x2, y2),
                   'ious': torch.tensor([]),
                   'masks': torch.empty(0, H, W, dtype=torch.uint8)}
            if return_indices: res['indices'] = torch.tensor([], dtype=torch.long)
            results.append(res)
            continue

        bbox_mask = torch.zeros((H, W), dtype=torch.uint8)
        bbox_mask[y1c:y2c, x1c:x2c] = 1

        inter = (masks_bin & bbox_mask).sum(dim=(1, 2)).to(torch.float32)
        union = (masks_bin | bbox_mask).sum(dim=(1, 2)).to(torch.float32)
        ious = torch.where(union > 0, inter/union, torch.zeros_like(union))

        keep = torch.nonzero(ious >= threshold, as_tuple=False).squeeze(-1)

        if keep.numel()>0:
            kept_ious = ious[keep]
            sort_idx = torch.argsort(kept_ious, descending=True)
            sel=keep[sort_idx][:topk]
            sel_ious = ious[sel]
            sel_masks = masks_bin[sel]
        else:
            sel=torch.tensor([], dtype=torch.long)
            sel_ious = torch.tensor([])
            sel_masks = torch.empty(0, H, W, dtype=torch.uint8)
        
        entry={
            'bbox': (x1, y1, x2, y2),
            'ious': sel_ious,
            'masks': sel_masks
        }
        if return_indices:
            entry['indices']=sel
        results.append(entry)
    return results

def postprocess(preds, img, orig_imgs, retina_masks=True, conf=0.25, iou=0.7, agnostic_nms=False):
    """TODO: filter by classes."""
    
    p = ops.non_max_suppression(preds[0],
                                conf,
                                iou,
                                agnostic_nms,
                                max_det=100,
                                nc=1)



    results = []
    proto = preds[1][-1] if len(preds[1]) == 3 else preds[1]  # second output is len 3 if pt, but only 1 if exported
    for i, pred in enumerate(p):
        orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        # path = self.batch[0]
        img_path = "ok"
        if not len(pred):  # save empty boxes
            results.append(Results(orig_img=orig_img, path=img_path, names="segment", boxes=pred[:, :6]))
            continue
        if retina_masks:
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto[i], pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        results.append(
            Results(orig_img=orig_img, path=img_path, names="1213", boxes=pred[:, :6], masks=masks))
    return results

def pre_processing(img_origin, imgsz=1024):
    h, w = img_origin.shape[:2]
    if h>w:
        scale   = min(imgsz / h, imgsz / w)
        inp     = np.zeros((imgsz, imgsz, 3), dtype = np.uint8)
        nw      = int(w * scale)
        nh      = int(h * scale)
        a = int((nh-nw)/2) 
        inp[: nh, a:a+nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    else:
        scale   = min(imgsz / h, imgsz / w)
        inp     = np.zeros((imgsz, imgsz, 3), dtype = np.uint8)
        nw      = int(w * scale)
        nh      = int(h * scale)
        a = int((nw-nh)/2) 

        inp[a: a+nh, :nw, :] = cv2.resize(cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB), (nw, nh))
    rgb = np.array([inp], dtype = np.float32) / 255.0
    return np.transpose(rgb, (0, 3, 1, 2))


class FastSam(object):
    def __init__(self, 
            model_weights = '/models/fastSAm_wrapper/fast_sam_1024.trt', 
            max_size = 1024):
        self.imgsz = (max_size, max_size)
        # Load model
        self.model = TrtModelNMS(model_weights, max_size)


    def segment(self, bgr_img, bboxes):
        ## Padded resize
        inp = pre_processing(bgr_img, self.imgsz[0])
        ## Inference
        t1 = time.time()
        print("[Input]: ", inp[0].transpose(0, 1, 2).shape)

        start = time.perf_counter()

        preds = self.model.run(inp)
        data_0 = torch.from_numpy(preds[5])
        data_1 = [[torch.from_numpy(preds[2]), torch.from_numpy(preds[3]), torch.from_numpy(preds[4])], torch.from_numpy(preds[1]), torch.from_numpy(preds[0])]
        preds = [data_0, data_1]

        result = postprocess(preds, inp, bgr_img)

        end = time.perf_counter()
        print(f"AIS exec.time: {(end-start)*1000:.2f} ms")
        
        if result[0].masks == None:
            print("no masks")
            return None
        else:
            masks = result[0].masks.data
            print("[Masks]: ", len(result), masks.shape)
 
        start = time.perf_counter()
        results = seleck_mask_by_bbox(masks, bboxes, threshold=0.3, topk=1)
        end = time.perf_counter()
        print(f"PGS exec.time: {(end-start)*1000:.2f} ms")
        # print(results)
 
        # image_with_masks = segment_everything(bgr_img, result, input_size=self.imgsz)
        # cv2.imwrite(f"/models/FastSam/outputs/obj_segment_trt.png", image_with_masks)
        
        return masks