from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DEMEncoderConfig:
    C: int = 256
    backbone_channels: Tuple[int, int, int, int] = (256, 512, 1024, 2048)
    down_ratio_f4: int = 16
    down_ratio_f5: int = 32
    use_learnable_gamma: bool = True
    init_gamma: float = 0.5
    lf_kernel: int = 7

    # torchvision detector uses these for normalization
    image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, d=1, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, dilation=d, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, act=True):
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, k=k, s=s, p=p, groups=in_ch, act=True)
        self.pw = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0, act=act)
    def forward(self, x):
        return self.pw(self.dw(x))

class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.fc1 = nn.Linear(ch, hidden)
        self.fc2 = nn.Linear(hidden, ch)
    def forward(self, x):
        b, c, _, _ = x.shape
        z = F.adaptive_avg_pool2d(x, 1).view(b, c)
        e = torch.sigmoid(self.fc2(F.relu(self.fc1(z), inplace=True))).view(b, c, 1, 1)
        return x * e

class LearnableGamma(nn.Module):
    def __init__(self, init=0.5, learnable=True):
        super().__init__()
        if learnable:
            self.gamma = nn.Parameter(torch.tensor(float(init)))
        else:
            self.register_buffer("gamma", torch.tensor(float(init)), persistent=False)
    def forward(self, x, att):
        return x * (1.0 + self.gamma * att)

# DEM_S
class GeoBlockSmall(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv1 = ConvBNAct(C, C, k=3, s=1, p=1, d=1)
        self.conv2 = ConvBNAct(C, C, k=3, s=1, p=2, d=2)
        self.conv_h = ConvBNAct(C, C, k=(1,3), s=1, p=(0,1))
        self.conv_v = ConvBNAct(C, C, k=(3,1), s=1, p=(1,0))
        self.fuse = ConvBNAct(3*C, C, k=1, s=1, p=0)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        xdir = self.conv_h(x1) + self.conv_v(x1)
        out = self.fuse(torch.cat([x2, xdir, x], dim=1))
        return out + x

class TextureBlockSmall(nn.Module):
    def __init__(self, C, gamma):
        super().__init__()
        self.dw3 = DepthwiseSeparableConv(C, C, k=3, s=1, p=1, act=True)
        self.dw5 = DepthwiseSeparableConv(C, C, k=5, s=1, p=2, act=True)
        self.fuse = ConvBNAct(2*C, C, k=1, s=1, p=0)
        self.rough_conv = nn.Conv2d(C, C, kernel_size=1, bias=True)
        self.gamma = gamma
    def forward(self, x):
        base = self.fuse(torch.cat([self.dw3(x), self.dw5(x)], dim=1))
        mu = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        rough = torch.sigmoid(self.rough_conv(torch.abs(x - mu)))
        return self.gamma(base, rough)

class DEMSmallScale(nn.Module):
    def __init__(self, C, gamma):
        super().__init__()
        self.geo = GeoBlockSmall(C)
        self.tex = TextureBlockSmall(C, gamma=gamma)
        self.fuse = ConvBNAct(2*C, C, k=1, s=1, p=0)
    def forward(self, x):
        return self.fuse(torch.cat([self.geo(x), self.tex(x)], dim=1))

# DEM_M
class GeoBlockMid(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.conv1 = ConvBNAct(C, C, k=3, s=1, p=1, d=1)
        self.conv2 = ConvBNAct(C, C, k=3, s=1, p=2, d=2)
        self.conv3 = ConvBNAct(C, C, k=3, s=1, p=3, d=3)
        self.fuse = ConvBNAct(4*C, C, k=1, s=1, p=0)
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        out = self.fuse(torch.cat([x1, x2, x3, x], dim=1))
        return out + x

class TextureBlockMid(nn.Module):
    def __init__(self, C, gamma):
        super().__init__()
        self.dw3 = DepthwiseSeparableConv(C, C, k=3, s=1, p=1, act=True)
        self.dw5 = DepthwiseSeparableConv(C, C, k=5, s=1, p=2, act=True)
        self.fuse = ConvBNAct(2*C, C, k=1, s=1, p=0)
        self.rough_conv = nn.Conv2d(C, C, kernel_size=1, bias=True)
        self.gamma = gamma
    def forward(self, x):
        base = self.fuse(torch.cat([self.dw3(x), self.dw5(x)], dim=1))
        mu = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        rough = torch.sigmoid(self.rough_conv(torch.abs(x - mu)))
        return self.gamma(base, rough)

class DEMMidScale(nn.Module):
    def __init__(self, C, gamma):
        super().__init__()
        self.geo = GeoBlockMid(C)
        self.tex = TextureBlockMid(C, gamma=gamma)
        self.fuse = ConvBNAct(2*C, C, k=1, s=1, p=0)
    def forward(self, x):
        return self.fuse(torch.cat([self.geo(x), self.tex(x)], dim=1))

# Lab conversion helpers (expects RGB in [0,1])
def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)

def rgb_to_lab_like(x_rgb: torch.Tensor) -> torch.Tensor:
    x_rgb = x_rgb.clamp(0.0, 1.0)
    r, g, b = x_rgb[:, 0:1], x_rgb[:, 1:2], x_rgb[:, 2:3]
    rgb_lin = torch.cat([_srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)], dim=1)
    M = x_rgb.new_tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = torch.einsum("bchw,dc->bdhw", rgb_lin, M)
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[:, 0:1] / Xn
    y = xyz[:, 1:2] / Yn
    z = xyz[:, 2:3] / Zn
    delta = 6.0 / 29.0
    def f(t):
        return torch.where(t > delta**3, t ** (1.0/3.0), (t / (3*delta**2)) + (4.0/29.0))
    fx, fy, fz = f(x), f(y), f(z)
    L = (116.0 * fy - 16.0) / 100.0
    a = 500.0 * (fx - fy) / 128.0
    b2 = 200.0 * (fy - fz) / 128.0
    return torch.cat([L, a, b2], dim=1)

# DEM_D
class DEMColorScale(nn.Module):
    def __init__(self, C, down_ratio, gamma, lf_kernel=7):
        super().__init__()
        self.down_ratio = int(down_ratio)
        self.color_embed = ConvBNAct(3, C, k=3, s=self.down_ratio, p=1)
        self.fuse = ConvBNAct(2*C, C, k=3, s=1, p=1)
        self.se = SEBlock(C, reduction=16)
        self.lf_conv = ConvBNAct(C, C, k=1, s=1, p=0)
        self.hf_conv1 = ConvBNAct(C, C, k=3, s=1, p=1)
        self.hf_conv2 = nn.Conv2d(C, C, kernel_size=1, bias=True)
        self.att_conv = nn.Conv2d(2*C, C, kernel_size=1, bias=True)
        self.lf_kernel = int(lf_kernel)
        self.gamma = gamma
    def forward(self, Ps, x_lab):
        Ls = self.color_embed(x_lab)
        if Ls.shape[-2:] != Ps.shape[-2:]:
            Ls = F.interpolate(Ls, size=Ps.shape[-2:], mode="bilinear", align_corners=False)
        C0 = self.fuse(torch.cat([Ps, Ls], dim=1))
        C1 = self.se(C0)
        k = self.lf_kernel
        pad = k // 2
        LF_pool = F.avg_pool2d(C1, kernel_size=k, stride=1, padding=pad)
        LF = self.lf_conv(LF_pool)
        HF = self.hf_conv2(self.hf_conv1(C1))
        A = torch.sigmoid(self.att_conv(torch.cat([LF, HF], dim=1)))
        return self.gamma(C1, A)

class FPNFuse(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.fpn4 = ConvBNAct(2*C, C, k=3, s=1, p=1)
        self.fpn3 = ConvBNAct(2*C, C, k=3, s=1, p=1)
        self.fpn2 = ConvBNAct(2*C, C, k=3, s=1, p=1)
    def forward(self, F2p, F3p, F4p, F5p):
        U5 = F5p
        U4 = self.fpn4(torch.cat([F4p, F.interpolate(U5, size=F4p.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        U3 = self.fpn3(torch.cat([F3p, F.interpolate(U4, size=F3p.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        U2 = self.fpn2(torch.cat([F2p, F.interpolate(U3, size=F2p.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        return U2, U3, U4, U5

class DEMVisionBackbone(nn.Module):
    """Backbone wrapper for torchvision detectors.

    IMPORTANT:
    - Torchvision FasterRCNN passes **normalized** tensors into the backbone.
    - For DEM_D (Lab), we first unnormalize to recover approximate RGB in [0,1].

    Output OrderedDict keys: '0','1','2','3' for MultiScaleRoIAlign.
    """
    def __init__(self, pyramid_backbone: nn.Module, cfg: DEMEncoderConfig,
                 disable_dem2=False, disable_dem3=False, disable_dem4=False, disable_dem5=False):
        super().__init__()
        self.body = pyramid_backbone
        self.cfg = cfg
        C = cfg.C
        C2, C3, C4, C5 = cfg.backbone_channels

        self.proj2 = ConvBNAct(C2, C, k=1, s=1, p=0)
        self.proj3 = ConvBNAct(C3, C, k=1, s=1, p=0)
        self.proj4 = ConvBNAct(C4, C, k=1, s=1, p=0)
        self.proj5 = ConvBNAct(C5, C, k=1, s=1, p=0)

        self.gamma_s = LearnableGamma(cfg.init_gamma, learnable=cfg.use_learnable_gamma)
        self.gamma_m = LearnableGamma(cfg.init_gamma, learnable=cfg.use_learnable_gamma)
        self.gamma_d4 = LearnableGamma(cfg.init_gamma, learnable=cfg.use_learnable_gamma)
        self.gamma_d5 = LearnableGamma(cfg.init_gamma, learnable=cfg.use_learnable_gamma)

        self.dem2 = DEMSmallScale(C, gamma=self.gamma_s)
        self.dem3 = DEMMidScale(C, gamma=self.gamma_m)
        self.dem4 = DEMColorScale(C, down_ratio=cfg.down_ratio_f4, gamma=self.gamma_d4, lf_kernel=cfg.lf_kernel)
        self.dem5 = DEMColorScale(C, down_ratio=cfg.down_ratio_f5, gamma=self.gamma_d5, lf_kernel=cfg.lf_kernel)

        self.disable_dem2 = bool(disable_dem2)
        self.disable_dem3 = bool(disable_dem3)
        self.disable_dem4 = bool(disable_dem4)
        self.disable_dem5 = bool(disable_dem5)

        self.fpn = FPNFuse(C)
        self.out_channels = C

        mean = torch.tensor(cfg.image_mean).view(1,3,1,1)
        std = torch.tensor(cfg.image_std).view(1,3,1,1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def _unnormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        # x_norm = (x - mean) / std  -> x = x_norm * std + mean
        return (x_norm * self._std + self._mean).clamp(0.0, 1.0)

    def forward(self, x_norm: torch.Tensor):
        # ResNet expects normalized input
        F2, F3, F4, F5 = self.body(x_norm)

        P2 = self.proj2(F2)
        P3 = self.proj3(F3)
        P4 = self.proj4(F4)
        P5 = self.proj5(F5)

        # Lab from (approx) RGB
        x_rgb = self._unnormalize(x_norm)
        x_lab = rgb_to_lab_like(x_rgb)

        F2p = P2 if self.disable_dem2 else self.dem2(P2)
        F3p = P3 if self.disable_dem3 else self.dem3(P3)
        F4p = P4 if self.disable_dem4 else self.dem4(P4, x_lab=x_lab)
        F5p = P5 if self.disable_dem5 else self.dem5(P5, x_lab=x_lab)

        U2, U3, U4, U5 = self.fpn(F2p, F3p, F4p, F5p)

        return OrderedDict([("0", U2), ("1", U3), ("2", U4), ("3", U5)])
