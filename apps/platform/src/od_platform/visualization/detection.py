#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : detection.py
# @Project   : ODPlatform / visualization
# @Function  : 与框架无关的检测结果值对象 (Detection)
"""绘制侧唯一认识的检测结果形态.

★ 为什么不用 ultralytics 的 Boxes: 美化模块要能被非 YOLO 的结果复用
  (传统 CV 检测器 / 第三方模型), 所以只吃"框 + 标签 + 置信度"这三样.
  从 ultralytics Results 到 Detection 的转换放在 BeautifyVisualizer.from_yolo_results().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Detection:
    """一个检测结果 (不可变).

    Attributes:
        xyxy:       左上右下坐标 (x1, y1, x2, y2), 像素.
        label:      类别名 (原始英文名或调用方自定义).
        confidence: 置信度 0~1; None 表示不显示置信度.
    """

    xyxy: tuple[float, float, float, float]
    label: str = ""
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if len(self.xyxy) != 4:
            raise ValueError(f"xyxy 必须是 4 元组, 收到: {self.xyxy!r}")
