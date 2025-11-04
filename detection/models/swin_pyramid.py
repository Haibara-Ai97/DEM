from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _load_torchvision_model(name: str, pretrained: bool) -> nn.Module:
    """Version-tolerant loader for torchvision models with either `weights=` or `pretrained=` API."""
    import torchvision

    if not hasattr(torchvision.models, name):
        raise ValueError(f"Unknown torchvision model: {name}")
    ctor = getattr(torchvision.models, name)

    weights_enum_map = {
        "swin_t": "Swin_T_Weights",
        "swin_s": "Swin_S_Weights",
        "swin_b": "Swin_B_Weights",
    }

    if pretrained:
        enum_name = weights_enum_map.get(name, None)
        if enum_name is not None and hasattr(torchvision.models, enum_name):
            enum = getattr(torchvision.models, enum_name)
            try:
                return ctor(weights=enum.DEFAULT)
            except Exception:
                pass

        try:
            return ctor(weights="DEFAULT")
        except Exception:
            pass
        try:
            return ctor(pretrained=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained {name}.") from e

    try:
        return ctor(weights=None)
    except Exception:
        return ctor(pretrained=False)


class SwinPyramidBackbone(nn.Module):
    """Swin backbone that returns 4 pyramid features (stage1..4).

    Torchvision Swin uses `model.features` with 8 blocks similar to:
      0: patch_embed
      1: stage1 (/4)
      2: patch_merging
      3: stage2 (/8)
      4: patch_merging
      5: stage3 (/16)
      6: patch_merging
      7: stage4 (/32)

    NOTE: intermediate tensors may be BHWC; we convert to BCHW before returning.
    """

    _CHANNELS = {
        "swin_t": (96, 192, 384, 768),
        "swin_s": (96, 192, 384, 768),
        "swin_b": (128, 256, 512, 1024),
    }

    def __init__(self, name: str = "swin_t", pretrained: bool = True) -> None:
        super().__init__()
        model = _load_torchvision_model(name=name, pretrained=pretrained)
        self.features: nn.Sequential = model.features  # type: ignore[attr-defined]
        if len(self.features) < 8:
            raise RuntimeError(
                f"Unexpected Swin features length={len(self.features)}. "
                "This wrapper assumes torchvision Swin layout with 8 feature blocks."
            )
        self.out_channels: Tuple[int, int, int, int] = self._CHANNELS.get(name, self._CHANNELS["swin_t"])

    @staticmethod
    def _to_bchw(x: torch.Tensor, expected_c: int) -> torch.Tensor:
        # If BCHW already
        if x.dim() == 4 and x.shape[1] == expected_c:
            return x
        # If BHWC
        if x.dim() == 4 and x.shape[-1] == expected_c:
            return x.permute(0, 3, 1, 2).contiguous()
        # Fallback: try BHWC->BCHW when last dim looks like channels
        if x.dim() == 4 and x.shape[-1] in (expected_c,):
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x: torch.Tensor):
        F2 = F3 = F4 = F5 = None
        c2, c3, c4, c5 = self.out_channels

        for i, m in enumerate(self.features):
            x = m(x)
            if i == 1:
                F2 = self._to_bchw(x, c2)
            elif i == 3:
                F3 = self._to_bchw(x, c3)
            elif i == 5:
                F4 = self._to_bchw(x, c4)
            elif i == 7:
                F5 = self._to_bchw(x, c5)

        if F2 is None or F3 is None or F4 is None or F5 is None:
            raise RuntimeError("Swin pyramid extraction failed; missing one or more stage features.")
        return F2, F3, F4, F5
