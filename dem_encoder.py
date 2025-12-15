from typing import Tuple
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DEMEncoderConfig:
    # Common channels after 1x1 projection
    C: int = 256
    # Output embedding dim for Adapter
    d_model: int = 256

    # Backbone channels for (F2,F3,F4,F5)
    backbone_channels: Tuple[int, int, int, int] = (64, 128, 256, 512)

    # Color branch downsample ratios to match H/16 and H/32
    down_ratio_f4: int = 16
    down_ratio_f5: int = 32

    # Roughness/attention amplification (can be constant or learnable; here: learnable)
    use_learnable_gamma: bool = True
    init_gamma: float = 0.5

    # LF pooling kernel for DEM_D (large kernel avg pooling)
    lf_kernel: int = 7

    # Whether to compute Lab internally if x_lab is None
    compute_lab_if_missing: bool = True
    # If computing Lab internally, assumes x_rgb is sRGB in [0,1]; if not, clamp
    clamp_rgb_01: bool = True


class ConvBNAct(nn.Module):
    """
    Simple Convolutional Neural Network
    Conv2d + BatchNorm2d + ReLU
    """
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 k: int | Tuple[int, int],
                 s: int = 1,
                 p: int | Tuple[int, int] = 1,
                 d: int = 1,
                 groups: int = 1,
                 act: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=k, stride=s, padding=p, dilation=d,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(output_channel)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable conv = depthwise (groups=input_channel) + pointwise(1 * 1)
    """
    def __init__(self,
                 input_channel: int,
                 output_channel: int,
                 k: int,
                 s: int = 1,
                 p: int = 0,
                 act: bool = True) -> None:
        super().__init__()
        self.dw = ConvBNAct(input_channel, input_channel, k=k, s=s, p=p, groups=input_channel, act=True)
        self.pw = ConvBNAct(input_channel, output_channel, k=1, s=1, p=0, groups=1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    """
    def __init__(self, ch: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(ch // reduction, 4)
        self.fc1 = nn.Linear(ch, hidden)
        self.fc2 = nn.Linear(hidden, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        z = F.adaptive_avg_pool2d(x, 1).view(b, c)
        e = torch.sigmoid(self.fc2(F.relu(self.fc1(z), inplace=True))).view(b, c, 1, 1)
        return x * e


class LearnableGamma(nn.Module):
    """
    y = x * (1 + gamma * att)
    gamma can be fixed or learnable
    """
    def __init__(self, init: float = 0.5, learnable: bool = True) -> None:
        super().__init__()
        if learnable:
            self.gamma = nn.Parameter(torch.tensor(float(init)))
        else:
            self.register_buffer("gamma", torch.tensor(float(init)), persistent=False)

    def forward(self, x: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + self.gamma * att)


class GeoBlockSmall(nn.Module):
    """
    GeoBlock_S:
      - 3x3 conv + 3x3 dilated(d=2)
      - directional conv: 1x3 and 3x1
      - concat([x2, xdir, x]) -> 1x1 -> residual add
    """
    def __init__(self, C: int):
        super().__init__()
        self.conv1 = ConvBNAct(C, C, k=3, s=1, p=1, d=1)
        self.conv2 = ConvBNAct(C, C, k=3, s=1, p=2, d=2)

        self.conv_h = ConvBNAct(C, C, k=(1, 3), s=1, p=(0, 1))
        self.conv_v = ConvBNAct(C, C, k=(3, 1), s=1, p=(1, 0))

        self.fuse = ConvBNAct(3*C, C, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        xdir = self.conv_h(x1) + self.conv_v(x1)
        out = self.fuse(torch.cat([x2, xdir, x], dim=1))
        return out + x


class TextureBlockSmall(nn.Module):
    """
    TextureBlock_S:
      - DSConv 3x3 and 5x5 -> concat -> 1x1 (texture base)
      - mu = avg_pool 3x3
      - rough = sigmoid(1x1(|x - mu|))
      - out = base * (1 + gamma * rough)
    """
    def __init__(self, C: int, gamma: LearnableGamma) -> None:
        super().__init__()
        self.dw3 = DepthwiseSeparableConv(C, C, k=3, s=1, p=1, act=True)
        self.dw5 = DepthwiseSeparableConv(C, C, k=5, s=1, p=2, act=True)
        self.fuse = ConvBNAct(2*C, C, k=1, s=1, p=0)

        self.rough_conv = nn.Conv2d(C, C, kernel_size=1, bias=True)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t3 = self.dw3(x)
        t5 = self.dw5(x)
        base = self.fuse(torch.cat([t3, t5], dim=1))

        mu = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        r = torch.abs(x - mu)
        rough = torch.sigmoid(self.rough_conv(r))

        return self.gamma(base, rough)


class DEMSmallScale(nn.Module):
    def __init__(self, C: int, gamma: LearnableGamma) -> None:
        super().__init__()
        self.geo = GeoBlockSmall(C)
        self.tex = TextureBlockSmall(C, gamma=gamma)
        self.fuse = ConvBNAct(2*C, C, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.geo(x)
        t = self.tex(x)
        return self.fuse(torch.cat([g, t], dim=1))


class GeoBlockMid(nn.Module):
    """
    GeoBlock_M:
      - 3 branches dilations (1,2,3) (implemented as conv on x1 per md)
      - concat([x1, x2, x3, x]) -> 1x1 -> residual add
    """
    def __init__(self, C: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(C, C, k=3, s=1, p=1, d=1)
        self.conv2 = ConvBNAct(C, C, k=3, s=1, p=2, d=2)
        self.conv3 = ConvBNAct(C, C, k=3, s=1, p=3, d=3)
        self.fuse = ConvBNAct(4 * C, C, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        out = self.fuse(torch.cat([x1, x2, x3, x], dim=1))
        return out + x


class TextureBlockMid(nn.Module):
    """
    TextureBlock_M:
      - same as small but mu uses 5x5 avg pool
    """
    def __init__(self, C: int, gamma: LearnableGamma) -> None:
        super().__init__()
        self.dw3 = DepthwiseSeparableConv(C, C, k=3, s=1, p=1, act=True)
        self.dw5 = DepthwiseSeparableConv(C, C, k=5, s=1, p=2, act=True)
        self.fuse = ConvBNAct(2 * C, C, k=1, s=1, p=0)

        self.rough_conv = nn.Conv2d(C, C, kernel_size=1, bias=True)
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t3 = self.dw3(x)
        t5 = self.dw5(x)
        base = self.fuse(torch.cat([t3, t5], dim=1))

        mu = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        r = torch.abs(x - mu)
        rough = torch.sigmoid(self.rough_conv(r))

        return self.gamma(base, rough)


class DEMMidScale(nn.Module):
    def __init__(self, C: int, gamma: LearnableGamma) -> None:
        super().__init__()
        self.geo = GeoBlockMid(C)
        self.tex = TextureBlockMid(C, gamma=gamma)
        self.fuse = ConvBNAct(2 * C, C, k=1, s=1, p=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.geo(x)
        t = self.tex(x)
        return self.fuse(torch.cat([g, t], dim=1))


def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    # x assumed in [0,1]
    return torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)


def rgb_to_lab_like(x_rgb: torch.Tensor, clamp_01: bool = True) -> torch.Tensor:
    """
    Convert sRGB [0,1] -> Lab (normalized roughly to ~[-1,1] range for stability):
      L in [0,1], a,b in [-1,1] approximately.

    If your training pipeline uses ImageNet normalization, DO NOT call this on normalized tensors.
    Prefer passing x_lab from Dataset.
    """
    if clamp_01:
        x_rgb = x_rgb.clamp(0.0, 1.0)

    r, g, b = x_rgb[:, 0:1], x_rgb[:, 1:2], x_rgb[:, 2:3]
    rgb_lin = torch.cat([_srgb_to_linear(r), _srgb_to_linear(g), _srgb_to_linear(b)], dim=1)

    # RGB -> XYZ (D65)
    M = x_rgb.new_tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])  # (3,3)

    xyz = torch.einsum("bchw,dc->bdhw", rgb_lin, M)  # (B,3,H,W)

    # Normalize by reference white
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[:, 0:1] / Xn
    y = xyz[:, 1:2] / Yn
    z = xyz[:, 2:3] / Zn

    delta = 6.0 / 29.0

    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > delta**3, t ** (1.0 / 3.0), (t / (3 * delta**2)) + (4.0 / 29.0))

    fx, fy, fz = f(x), f(y), f(z)

    L = (116.0 * fy - 16.0) / 100.0          # ~[0,1]
    a = 500.0 * (fx - fy) / 128.0            # ~[-1,1]
    b2 = 200.0 * (fy - fz) / 128.0           # ~[-1,1]

    return torch.cat([L, a, b2], dim=1)


class DEMColorScale(nn.Module):
    def __init__(self,
                 C: int,
                 down_ratio: int,
                 gamma: LearnableGamma,
                 lf_kernel: int = 7) -> None:
        super().__init__()
        self.down_ratio = down_ratio
        self.color_embed = ConvBNAct(3, C, k=3, s=self.down_ratio, p=1)
        self.fuse = ConvBNAct(2*C, C, k=3, s=1, p=1)

        self.se = SEBlock(C, reduction=16)
        self.lf_conv = ConvBNAct(C, C, k=1, s=1, p=0)
        self.hf_conv1 = ConvBNAct(C, C, k=3, s=1, p=1)
        self.hf_conv2 = nn.Conv2d(C, C, kernel_size=1, bias=True)

        self.att_conv = nn.Conv2d(2*C, C, kernel_size=1, bias=True)

        self.lf_kernel = int(lf_kernel)
        self.gamma = gamma

    def forward(self, Ps: torch.Tensor, x_lab: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, C: int) -> None:
        super().__init__()
        self.fpn4 = ConvBNAct(2*C, C, k=3, s=1, p=1)
        self.fpn3 = ConvBNAct(2*C, C, k=3, s=1, p=1)
        self.fpn2 = ConvBNAct(2*C, C, k=3, s=1, p=1)

    def forward(self, F2p: torch.Tensor, F3p: torch.Tensor, F4p: torch.Tensor, F5p: torch.Tensor) -> torch.Tensor:
        U5 = F5p
        U4 = self.fpn4(torch.cat([F4p, F.interpolate(U5, size=F4p.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        U3 = self.fpn3(
            torch.cat([F3p, F.interpolate(U4, size=F3p.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        U2 = self.fpn2(
            torch.cat([F2p, F.interpolate(U3, size=F2p.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        return U2


class DEMVisionEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, cfg: DEMEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone

        C2, C3, C4, C5 = cfg.backbone_channels
        C = cfg.C

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

        self.fpn = FPNFuse(C)

        self.out_proj = ConvBNAct(C, cfg.d_model, k=1, s=1, p=0, act=False)

    def forward(self,
                x_rgb: torch.Tensor,
                *,
                x_lab: Optional[torch.Tensor] = None,
                return_intermediate: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]] | torch.Tensor:
        feats = self.backbone(x_rgb)
        if not isinstance(feats, (tuple, list)) or len(feats) != 4:
            raise ValueError("Backbone must return (F2,F3,F4,F5) with 4 feature maps")
        F2, F3, F4, F5 = feats

        P2 = self.proj2(F2)
        P3 = self.proj3(F3)
        P4 = self.proj4(F4)
        P5 = self.proj5(F5)

        if x_lab is None:
            if not self.cfg.compute_lab_if_missing:
                raise ValueError("x_lab is required when compute_lab_if_missing is False")
            x_lab = rgb_to_lab_like(x_rgb, clamp_01=self.cfg.clamp_rgb_01)

        F2p = self.dem2(P2)
        F3p = self.dem3(P3)
        F4p = self.dem4(P4, x_lab=x_lab)
        F5p = self.dem5(P5, x_lab=x_lab)

        F_def = self.fpn(F2p, F3p, F4p, F5p)

        F_enc = self.out_proj(F_def)

        if not return_intermediate:
            return F_enc

        pack: Dict[str, Any] = {
            "F2": F2, "F3": F3, "F4": F4, "F5": F5,
            "P2": P2, "P3": P3, "P4": P4, "P5": P5,
            "F2p": F2p, "F3p": F3p, "F4p": F4p, "F5p": F5p,
            "F_def": F_def,
        }
        return F_enc, pack

    @staticmethod
    def to_tokens(F_enc: torch.Tensor) -> torch.Tensor:
        b, d, h, w = F_enc.shape
        return F_enc.view(b, d, h * w).permute(0, 2, 1).contiguous()


class DummyPyramidBackbone(nn.Module):
    """
    Simple CNN pyramid backbone for debugging only.
    Produces F2..F5 at /4,/8,/16,/32.
    """
    def __init__(self, out_channels: Sequence[int] = (64, 128, 256, 512)) -> None:
        super().__init__()
        c2, c3, c4, c5 = out_channels

        # stem: /2
        self.stem = nn.Sequential(
            ConvBNAct(3, c2, k=3, s=2, p=1),
            ConvBNAct(c2, c2, k=3, s=1, p=1),
        )
        # stage2: /4
        self.s2 = nn.Sequential(
            ConvBNAct(c2, c2, k=3, s=2, p=1),
            ConvBNAct(c2, c2, k=3, s=1, p=1),
        )
        # stage3: /8
        self.s3 = nn.Sequential(
            ConvBNAct(c2, c3, k=3, s=2, p=1),
            ConvBNAct(c3, c3, k=3, s=1, p=1),
        )
        # stage4: /16
        self.s4 = nn.Sequential(
            ConvBNAct(c3, c4, k=3, s=2, p=1),
            ConvBNAct(c4, c4, k=3, s=1, p=1),
        )
        # stage5: /32
        self.s5 = nn.Sequential(
            ConvBNAct(c4, c5, k=3, s=2, p=1),
            ConvBNAct(c5, c5, k=3, s=1, p=1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)    # /2
        F2 = self.s2(x)     # /4
        F3 = self.s3(F2)    # /8
        F4 = self.s4(F3)    # /16
        F5 = self.s5(F4)    # /32
        return F2, F3, F4, F5


def smoke_test() -> None:
    cfg = DEMEncoderConfig(
        C=256,
        d_model=256,
        backbone_channels=(64, 128, 256, 512),
        down_ratio_f4=16,
        down_ratio_f5=32,
        use_learnable_gamma=True,
        init_gamma=0.5,
    )
    backbone = DummyPyramidBackbone(out_channels=cfg.backbone_channels)
    model = DEMVisionEncoder(backbone, cfg).eval()

    x = torch.rand(2, 3, 512, 512)
    with torch.no_grad():
        F_enc, pack = model(x, return_intermediate=True)
        assert F_enc.shape == (2, cfg.d_model, 128, 128), F_enc.shape
        toks = model.to_tokens(F_enc)
        assert toks.shape == (2, 128 * 128, cfg.d_model), toks.shape
    print("Smoke test passwd:", F_enc.shape, toks.shape)


if __name__ == "__main__":
    smoke_test()














