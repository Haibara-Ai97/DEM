from __future__ import annotations

from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: int = 0, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class FPNFuse(nn.Module):
    """Simple top-down FPN fuse (4 levels)."""

    def __init__(self, C: int):
        super().__init__()
        self.fpn4 = ConvBNAct(2 * C, C, k=3, s=1, p=1)
        self.fpn3 = ConvBNAct(2 * C, C, k=3, s=1, p=1)
        self.fpn2 = ConvBNAct(2 * C, C, k=3, s=1, p=1)

    def forward(self, F2: torch.Tensor, F3: torch.Tensor, F4: torch.Tensor, F5: torch.Tensor):
        U5 = F5
        U4 = self.fpn4(torch.cat([F4, F.interpolate(U5, size=F4.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        U3 = self.fpn3(torch.cat([F3, F.interpolate(U4, size=F3.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        U2 = self.fpn2(torch.cat([F2, F.interpolate(U3, size=F2.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        return U2, U3, U4, U5


class PyramidFPNBackbone(nn.Module):
    """Wrap a pyramid body (returns 4 feature maps) into a Faster R-CNN compatible backbone.

    Torchvision FasterRCNN requirement:
      - forward(x) returns an OrderedDict[str, Tensor]
      - attribute out_channels is the channel dim of each output level
    """

    def __init__(self, pyramid_body: nn.Module, in_channels: Tuple[int, int, int, int], out_channels: int = 256):
        super().__init__()
        self.body = pyramid_body
        c2, c3, c4, c5 = in_channels
        C = int(out_channels)

        self.proj2 = ConvBNAct(int(c2), C, k=1, s=1, p=0)
        self.proj3 = ConvBNAct(int(c3), C, k=1, s=1, p=0)
        self.proj4 = ConvBNAct(int(c4), C, k=1, s=1, p=0)
        self.proj5 = ConvBNAct(int(c5), C, k=1, s=1, p=0)

        self.fpn = FPNFuse(C)
        self.out_channels = C

    def forward(self, x: torch.Tensor):
        F2, F3, F4, F5 = self.body(x)
        P2 = self.proj2(F2)
        P3 = self.proj3(F3)
        P4 = self.proj4(F4)
        P5 = self.proj5(F5)
        U2, U3, U4, U5 = self.fpn(P2, P3, P4, P5)
        return OrderedDict([("0", U2), ("1", U3), ("2", U4), ("3", U5)])
