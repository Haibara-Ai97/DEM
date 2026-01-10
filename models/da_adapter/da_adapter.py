# models/da_adapter.py
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class DAAdapterConfig:
    in_channels: int
    llm_dim: int
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    use_layernorm: bool = True
    use_fp32_ln: bool = True


class TokenResidualMLP(nn.Module):
    """
    per-token 残差 MLP：x = x + MLP(LN(x))
    """
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 use_layernorm: bool = True, use_fp32_ln: bool = True):
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
    """
    输入：feature map (B,C,H,W)
    输出：tokens (B,N,llm_dim)
      - Conv1x1 投影到 llm_dim
      - per-token 残差 MLP
      - 输出 LN
    """
    def __init__(self, cfg: DAAdapterConfig):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Conv2d(cfg.in_channels, cfg.llm_dim, kernel_size=1, bias=True)
        self.mlp = TokenResidualMLP(cfg.llm_dim, cfg.mlp_ratio, cfg.dropout, cfg.use_layernorm, cfg.use_fp32_ln)
        self.out_ln = nn.LayerNorm(cfg.llm_dim) if cfg.use_layernorm else nn.Identity()

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        # feat: (B,C,H,W)
        x = self.proj(feat)                                  # (B,D,H,W)
        x = x.flatten(2).transpose(1, 2).contiguous()        # (B,N,D)
        x = self.mlp(x)
        if self.cfg.use_layernorm:
            if self.cfg.use_fp32_ln and x.dtype in (torch.float16, torch.bfloat16):
                x = self.out_ln(x.float()).to(x.dtype)
            else:
                x = self.out_ln(x)
        return x
