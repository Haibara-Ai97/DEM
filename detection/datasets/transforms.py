from __future__ import annotations
from typing import Dict, Any

import torch
import torchvision.transforms.functional as F
from PIL import Image

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image: Image.Image, target: Dict[str, Any]):
        # float tensor in [0,1]
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p
    def __call__(self, image: torch.Tensor, target: Dict[str, Any]):
        if torch.rand(1).item() < self.p:
            image = torch.flip(image, dims=[2])  # flip width (CHW -> dim=2)
            w = image.shape[2]
            boxes = target["boxes"]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return image, target
