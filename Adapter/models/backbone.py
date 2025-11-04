from __future__ import annotations
import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from typing import Tuple, Dict


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SimplePyramidBackbone(nn.Module):
    """
    输出：
      F2: (B,256,H/4,W/4)
      F3: (B,512,H/8,W/8)
      F4: (B,1024,H/16,W/16)
      F5: (B,2048,H/32,W/32)
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # F2
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 128, k=3, s=1, p=1),
            ConvBNReLU(128, 256, k=3, s=1, p=1),
        )
        # F3
        self.down3 = ConvBNReLU(256, 512, k=3, s=2, p=1)
        self.stage3 = nn.Sequential(
            ConvBNReLU(512, 512, k=3, s=1, p=1),
            ConvBNReLU(512, 512, k=3, s=1, p=1),
        )
        # F4
        self.down4 = ConvBNReLU(512, 1024, k=3, s=2, p=1)
        self.stage4 = nn.Sequential(
            ConvBNReLU(1024, 1024, k=3, s=1, p=1),
            ConvBNReLU(1024, 1024, k=3, s=1, p=1),
        )
        # F5
        self.down5 = ConvBNReLU(1024, 2048, k=3, s=2, p=1)
        self.stage5 = nn.Sequential(
            ConvBNReLU(2048, 2048, k=3, s=1, p=1),
            ConvBNReLU(2048, 2048, k=3, s=1, p=1),
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        f2 = self.stage2(x)
        f3 = self.stage3(self.down3(f2))
        f4 = self.stage4(self.down4(f3))
        f5 = self.stage5(self.down5(f4))
        return f2, f3, f4, f5


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
        self.out_channels = (256, 512, 1024, 2048)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        outs: Dict[str, torch.Tensor] = self.body(x)
        return outs["F2"], outs["F3"], outs["F4"], outs["F5"]