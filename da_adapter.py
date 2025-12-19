from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DAAdapterConfig:
    in_channels: int
    llm_dim: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_layernorm: bool = True
    use_fp32_ln: bool = True
    output_2d: bool = False


class TokenResidualMLP(nn.Module):
    def __init__(self,
                 dim: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.0,
                 use_layernorm: bool = True,
                 use_fp32_ln: bool = True):
        super().__init__()
        hidden = int(dim * mlp_ratio)

        self.use_layernorm = use_layernorm
        self.use_fp32_ln = use_fp32_ln
        self.ln = nn.LayerNorm(dim) if use_layernorm else nn.Identity()

        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.use_layernorm:
            if self.use_fp32_ln and x.dtype in (torch.float16, torch.bfloat16):
                x = self.ln(x.float()).to(residual.dtype)
            else:
                x = self.ln(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return residual + x


class DAAdapter(nn.Module):
    def __init__(self,
                 cfg: DAAdapterConfig):
        super().__init__()
        self.cfg = cfg

        self.proj_conv = nn.Conv2d(cfg.in_channels, cfg.llm_dim, kernel_size=1, bias=True)
        self.proj_linear = nn.Linear(cfg.in_channels, cfg.llm_dim, bias=True)

        self.token_mlp = TokenResidualMLP(
            dim=cfg.llm_dim,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
            use_layernorm=cfg.use_layernorm,
            use_fp32_ln=cfg.use_fp32_ln
        )

        self.out_ln = nn.LayerNorm(cfg.llm_dim) if cfg.use_layernorm else nn.Identity()

    def forward(self,
                x: torch.Tensor,
                *,
                return_hw: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[int, int]]]:
        if x.dim() == 4:
            b, c, h, w = x.shape
            y = self.proj_conv(x)
            y = y.flatten(2).transpose(1, 2).contiguous()
        elif x.dim() == 3:
            b, n, c = x.shape
            y = self.proj_linear(x)
            h, w = -1, -1
        else:
            raise ValueError(f"DAAdapter expects 3D or 4D input, got shape {tuple(x.shape)}")

        y = self.token_mlp(y)

        if self.cfg.use_layernorm:
            if self.cfg.use_fp32_ln and y.dtype in (torch.float16, torch.bfloat16):
                y = self.out_ln(y.float()).to(y.dtype)
            else:
                y = self.out_ln(y)

        if self.cfg.output_2d and x.dim() == 4:
            y2d = y.transpose(1, 2).reshape(b, self.cfg.llm_dim, h, w).contiguous()
            return (y2d, (h, w)) if return_hw else y2d

        return (y, (h, w)) if return_hw else y


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)