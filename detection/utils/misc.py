from __future__ import annotations
import os
import random
from typing import Any, Dict

import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    lr_scheduler: Any, epoch: int, best_map: float, extra: Dict[str, Any] | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "epoch": epoch,
        "best_map": best_map,
    }
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, path)

def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None,
                    lr_scheduler: Any | None = None, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if lr_scheduler is not None and ckpt.get("lr_scheduler") is not None:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    return ckpt
