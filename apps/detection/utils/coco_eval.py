from __future__ import annotations

from typing import Dict, List, Any, Optional
import torch

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception:
    COCO = None
    COCOeval = None

def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=1)

def coco_from_dict(coco_dict: Dict[str, Any]) -> Any:
    if COCO is None:
        raise ImportError("pycocotools is required. pip install pycocotools")
    coco = COCO()
    coco.dataset = coco_dict
    coco.createIndex()
    return coco

class CocoEvaluator:
    """COCO evaluator for bbox, supports GT from json file or from in-memory dict."""
    def __init__(self, ann_json: Optional[str] = None, ann_dict: Optional[Dict[str, Any]] = None, iou_type: str = "bbox") -> None:
        if COCO is None or COCOeval is None:
            raise ImportError("pycocotools is required for COCO evaluation. Please pip install pycocotools.")
        if (ann_json is None) == (ann_dict is None):
            raise ValueError("Provide exactly one of ann_json or ann_dict.")
        self.coco_gt = COCO(ann_json) if ann_json is not None else coco_from_dict(ann_dict)
        self.iou_type = iou_type
        self.img_ids: List[int] = []
        self.predictions: List[dict] = []

    @torch.inference_mode()
    def update(self, outputs: List[Dict[str, torch.Tensor]], targets: List[Dict[str, Any]]):
        for out, tgt in zip(outputs, targets):
            image_id = int(tgt["image_id"].item()) if tgt["image_id"].numel() == 1 else int(tgt["image_id"][0].item())
            self.img_ids.append(image_id)
            if out is None:
                continue
            boxes = out["boxes"]
            scores = out["scores"]
            labels = out["labels"]
            if boxes.numel() == 0:
                continue
            boxes_xywh = xyxy_to_xywh(boxes).cpu()
            scores = scores.cpu()
            labels = labels.cpu()
            for b, s, l in zip(boxes_xywh, scores, labels):
                self.predictions.append({
                    "image_id": image_id,
                    "category_id": int(l.item()),
                    "bbox": [float(v) for v in b.tolist()],
                    "score": float(s.item()),
                })

    def summarize(self) -> Dict[str, float]:
        img_ids = list(sorted(set(self.img_ids)))
        if len(self.predictions) == 0:
            return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0}

        coco_dt = self.coco_gt.loadRes(self.predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, self.iou_type)
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats = coco_eval.stats
        return {"mAP": float(stats[0]), "mAP50": float(stats[1]), "mAP75": float(stats[2])}
