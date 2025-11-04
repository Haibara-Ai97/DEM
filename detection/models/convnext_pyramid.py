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

    # Prefer weights enums if available (newer torchvision)
    weights_enum_map = {
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
        "convnext_small": "ConvNeXt_Small_Weights",
        "convnext_base": "ConvNeXt_Base_Weights",
        "convnext_large": "ConvNeXt_Large_Weights",
    }

    if pretrained:
        enum_name = weights_enum_map.get(name, None)
        if enum_name is not None and hasattr(torchvision.models, enum_name):
            enum = getattr(torchvision.models, enum_name)
            try:
                return ctor(weights=enum.DEFAULT)
            except Exception:
                pass

        # Fallbacks
        try:
            return ctor(weights="DEFAULT")  # some versions accept this
        except Exception:
            pass
        try:
            return ctor(pretrained=True)  # older API
        except Exception as e:
            raise RuntimeError(f"Failed to load pretrained {name}.") from e

    # not pretrained
    try:
        return ctor(weights=None)
    except Exception:
        return ctor(pretrained=False)


class ConvNeXtPyramidBackbone(nn.Module):
    """ConvNeXt backbone that returns 4 pyramid features (stage1..4).

    Typical torchvision ConvNeXt layout:
      features[0] : stem (stride=4)
      features[1] : stage1 blocks  (/4)
      features[2] : downsample
      features[3] : stage2 blocks  (/8)
      features[4] : downsample
      features[5] : stage3 blocks  (/16)
      features[6] : downsample
      features[7] : stage4 blocks  (/32)

    We capture outputs after indices 1, 3, 5, 7.
    """

    _CHANNELS = {
        "convnext_tiny": (96, 192, 384, 768),
        "convnext_small": (96, 192, 384, 768),
        "convnext_base": (128, 256, 512, 1024),
        "convnext_large": (192, 384, 768, 1536),
    }

    def __init__(self, name: str = "convnext_tiny", pretrained: bool = True) -> None:
        super().__init__()
        model = _load_torchvision_model(name=name, pretrained=pretrained)
        self.features: nn.Sequential = model.features  # type: ignore[attr-defined]
        if len(self.features) < 8:
            raise RuntimeError(
                f"Unexpected ConvNeXt features length={len(self.features)}. "
                "This wrapper assumes torchvision ConvNeXt layout with 8 feature blocks."
            )
        self.out_channels: Tuple[int, int, int, int] = self._CHANNELS.get(name, self._CHANNELS["convnext_tiny"])

    def forward(self, x: torch.Tensor):
        F2 = F3 = F4 = F5 = None
        for i, m in enumerate(self.features):
            x = m(x)
            if i == 1:
                F2 = x
            elif i == 3:
                F3 = x
            elif i == 5:
                F4 = x
            elif i == 7:
                F5 = x
        if F2 is None or F3 is None or F4 is None or F5 is None:
            raise RuntimeError("ConvNeXt pyramid extraction failed; missing one or more stage features.")
        return F2, F3, F4, F5
