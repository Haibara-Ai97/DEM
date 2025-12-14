from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        b, c, _ = x.shape
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
