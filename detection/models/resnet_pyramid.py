from __future__ import annotations
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

class ResNetPyramidBackbone(nn.Module):
    """ResNet backbone that returns pyramid features (C2..C5).

    Output:
      F2: /4  (layer1)
      F3: /8  (layer2)
      F4: /16 (layer3)
      F5: /32 (layer4)
    """
    def __init__(self, name: str = "resnet50", pretrained: bool = True) -> None:
        super().__init__()
        if not hasattr(torchvision.models, name):
            raise ValueError(f"Unknown resnet name: {name}")
        resnet = getattr(torchvision.models, name)(weights="DEFAULT" if pretrained else None)
        self.body = IntermediateLayerGetter(resnet, return_layers={
            "layer1": "F2",
            "layer2": "F3",
            "layer3": "F4",
            "layer4": "F5",
        })
        # resnet50: 256,512,1024,2048
        self.out_channels = (256, 512, 1024, 2048)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outs: Dict[str, torch.Tensor] = self.body(x)
        return outs["F2"], outs["F3"], outs["F4"], outs["F5"]
