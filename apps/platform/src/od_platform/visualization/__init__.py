#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Project   : ODPlatform / visualization
# @Function  : visualization 包对外面板
"""visualization — 推理结果美化绘制.

业务方按需引入:
    from od_platform.visualization import BeautifyVisualizer, DrawStyle, Detection

典型用法 (D7 InferService):
    viz = BeautifyVisualizer(labels=model.names.values(),
                             label_mapping={"person": "人员"},
                             color_mapping={"person": [0, 255, 0]})
    dets = BeautifyVisualizer.from_yolo_results(boxes, confidences, labels)
    annotated = viz.draw(frame.image, dets, style=DrawStyle.from_image_size(h, w))

★ 不导出内部绘制分支 (_draw_with_cv2 / _draw_with_pil) —— 实现细节.
"""
from .detection import Detection
from .style import Color, DrawStyle, to_bgr_tuple
from .visualizer import BeautifyVisualizer
from .segmentation import MaskStyle, MaskVisualizer

__all__ = [
    "BeautifyVisualizer",
    "DrawStyle",
    "Detection",
    "Color",
    "to_bgr_tuple",
    "MaskStyle",
    "MaskVisualizer",
]
