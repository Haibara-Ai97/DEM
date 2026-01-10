from __future__ import annotations

from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .coco_eval import CocoEvaluator

def train_one_epoch(model, optimizer, data_loader: DataLoader, device: torch.device,
                    epoch: int, lr_scheduler=None, print_freq: int = 20):
    model.train()
    losses_epoch = 0.0
    n = 0
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Train E{epoch}", ncols=110)
    for it, (images, targets) in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # torchvision detector returns dict of losses in training
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_item = float(losses.detach().item())
        losses_epoch += loss_item
        n += 1

        if it % print_freq == 0:
            pbar.set_postfix({"loss": f"{loss_item:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

    return {"loss": losses_epoch / max(n, 1)}

@torch.inference_mode()
def evaluate(model, data_loader: DataLoader, device: torch.device, *, ann_json: Optional[str] = None, ann_dict: Optional[Dict[str, Any]] = None):
    model.eval()
    evaluator = CocoEvaluator(ann_json=ann_json, ann_dict=ann_dict, iou_type="bbox")

    pbar = tqdm(data_loader, total=len(data_loader), desc="Eval", ncols=110)
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        outputs = model(images)  # list of dicts
        evaluator.update(outputs, targets)

    stats = evaluator.summarize()
    return stats
