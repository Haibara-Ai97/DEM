from __future__ import annotations

from typing import List
import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from .resnet_pyramid import ResNetPyramidBackbone
from .dem_encoder import DEMVisionBackbone, DEMEncoderConfig

def build_baseline_fasterrcnn(num_classes_fg: int) -> FasterRCNN:
    num_classes = int(num_classes_fg) + 1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

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

    # Anchors per pyramid level (4 levels)
    sizes = tuple((int(s),) for s in anchor_sizes)
    ratios = tuple(tuple(float(r) for r in aspect_ratios) for _ in range(len(sizes)))
    anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)

    roi_pooler = MultiScaleRoIAlign(featmap_names=["0","1","2","3"], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
    )
    return model
