from __future__ import annotations

from typing import List

import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from .resnet_pyramid import ResNetPyramidBackbone
from models.dem_encoder import DEMVisionBackbone, DEMEncoderConfig
from .pyramid_fpn_backbone import PyramidFPNBackbone
from .convnext_pyramid import ConvNeXtPyramidBackbone
from .swin_pyramid import SwinPyramidBackbone


def _build_frcnn_with_backbone(
    backbone: torchvision.nn.Module,
    num_classes: int,
    anchor_sizes: List[int],
    aspect_ratios: List[float],
) -> FasterRCNN:
    # 4 pyramid levels -> 4 anchor size groups
    sizes = tuple((int(s),) for s in anchor_sizes)
    ratios = tuple(tuple(float(r) for r in aspect_ratios) for _ in range(len(sizes)))
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)

    roi_pooler = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone=backbone,
        num_classes=int(num_classes),
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    return model


def build_baseline_fasterrcnn(num_classes_fg: int) -> FasterRCNN:
    """Torchvision official Faster R-CNN baseline (note: internal backbone+FPN+anchors differ from our custom FPN)."""
    num_classes = int(num_classes_fg) + 1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model


def build_r50_custom_fasterrcnn(
    num_classes_fg: int,
    anchor_sizes: List[int] = [16, 32, 64, 128],
    aspect_ratios: List[float] = [0.5, 1.0, 2.0],
    resnet_name: str = "resnet50",
    pretrained_resnet: bool = True,
    fpn_C: int = 256,
) -> FasterRCNN:
    """Fixed detector framework, only encoder=ResNet pyramid (custom projection+FPN)."""
    num_classes = int(num_classes_fg) + 1
    pyramid = ResNetPyramidBackbone(name=resnet_name, pretrained=pretrained_resnet)
    backbone = PyramidFPNBackbone(pyramid_body=pyramid, in_channels=pyramid.out_channels, out_channels=fpn_C)
    return _build_frcnn_with_backbone(backbone, num_classes, anchor_sizes, aspect_ratios)


def build_convnext_fasterrcnn(
    num_classes_fg: int,
    anchor_sizes: List[int] = [16, 32, 64, 128],
    aspect_ratios: List[float] = [0.5, 1.0, 2.0],
    convnext_name: str = "convnext_tiny",
    pretrained_convnext: bool = True,
    fpn_C: int = 256,
) -> FasterRCNN:
    """Fixed detector framework, only encoder=ConvNeXt pyramid (custom projection+FPN)."""
    num_classes = int(num_classes_fg) + 1
    pyramid = ConvNeXtPyramidBackbone(name=convnext_name, pretrained=pretrained_convnext)
    backbone = PyramidFPNBackbone(pyramid_body=pyramid, in_channels=pyramid.out_channels, out_channels=fpn_C)
    return _build_frcnn_with_backbone(backbone, num_classes, anchor_sizes, aspect_ratios)


def build_swin_fasterrcnn(
    num_classes_fg: int,
    anchor_sizes: List[int] = [16, 32, 64, 128],
    aspect_ratios: List[float] = [0.5, 1.0, 2.0],
    swin_name: str = "swin_t",
    pretrained_swin: bool = True,
    fpn_C: int = 256,
) -> FasterRCNN:
    """Fixed detector framework, only encoder=Swin pyramid (custom projection+FPN)."""
    num_classes = int(num_classes_fg) + 1
    pyramid = SwinPyramidBackbone(name=swin_name, pretrained=pretrained_swin)
    backbone = PyramidFPNBackbone(pyramid_body=pyramid, in_channels=pyramid.out_channels, out_channels=fpn_C)
    return _build_frcnn_with_backbone(backbone, num_classes, anchor_sizes, aspect_ratios)


def build_dem_fasterrcnn(
    num_classes_fg: int,
    dem_C: int = 256,
    init_gamma: float = 0.5,
    lf_kernel: int = 7,
    anchor_sizes: List[int] = [16, 32, 64, 128],
    aspect_ratios: List[float] = [0.5, 1.0, 2.0],
    disable_dem2: bool = False,
    disable_dem3: bool = False,
    disable_dem4: bool = False,
    disable_dem5: bool = False,
    resnet_name: str = "resnet50",
    pretrained_resnet: bool = True,
) -> FasterRCNN:
    """Your DEM-Encoder backbone (as provided earlier)."""
    num_classes = int(num_classes_fg) + 1

    pyramid = ResNetPyramidBackbone(name=resnet_name, pretrained=pretrained_resnet)
    cfg = DEMEncoderConfig(
        C=dem_C,
        backbone_channels=pyramid.out_channels,
        down_ratio_f4=16,
        down_ratio_f5=32,
        init_gamma=init_gamma,
        lf_kernel=lf_kernel,
    )
    backbone = DEMVisionBackbone(
        pyramid_backbone=pyramid,
        cfg=cfg,
        disable_dem2=disable_dem2,
        disable_dem3=disable_dem3,
        disable_dem4=disable_dem4,
        disable_dem5=disable_dem5,
    )

    return _build_frcnn_with_backbone(backbone, num_classes, anchor_sizes, aspect_ratios)
