#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : segmentation.py
# @Project   : ODPlatform / visualization
# @Function  : 语义掩膜叠加、热力图和 GT 对比
"""只吃图像与概率图,不依赖 torch 或具体模型。概率必须是 sigmoid 后的 0~1。"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .style import Color, to_bgr_tuple


@dataclass(frozen=True)
class MaskStyle:
    alpha: float = 0.45
    threshold: float = 0.5
    color: Color = (0, 255, 0)
    contour_thickness: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.alpha <= 1 or not 0 < self.threshold < 1:
            raise ValueError("alpha 需要在 0~1, threshold 需要在 0~1 开区间")
        if self.contour_thickness < 0:
            raise ValueError("contour_thickness 不能为负")
        if any(channel < 0 or channel > 255 for channel in to_bgr_tuple(self.color)):
            raise ValueError("颜色通道需要在 0~255")


class MaskVisualizer:
    """D7 语义分割绘制器。输入图像不会被原地修改。"""

    def __init__(self, style: MaskStyle | None = None) -> None:
        self.style = style or MaskStyle()

    def draw(self, image: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image 必须是 HxWx3 uint8 BGR 图像")
        self._validate_probabilities(probabilities, image.shape[:2])
        mask = probabilities >= self.style.threshold
        output = image.copy()
        color = np.asarray(to_bgr_tuple(self.style.color), dtype=np.float32)
        output[mask] = np.rint((1 - self.style.alpha) * output[mask] + self.style.alpha * color).astype(np.uint8)
        if self.style.contour_thickness and mask.any():
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, to_bgr_tuple(self.style.color), self.style.contour_thickness)
        return output

    def comparison(self, image: np.ndarray, probabilities: np.ndarray,
                   truth: np.ndarray | None = None) -> np.ndarray:
        """无 GT: 原图/概率/叠加。有 GT: 原图/GT/预测/像素误差。"""
        overlay = self.draw(image, probabilities)
        if truth is None:
            heatmap = cv2.applyColorMap(np.rint(probabilities * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            panels = [("Original", image), ("Probability", heatmap), ("Prediction", overlay)]
        else:
            if truth.shape != probabilities.shape:
                raise ValueError("GT 与预测尺寸不一致")
            positive = probabilities >= self.style.threshold
            target = truth > 0
            error = image.copy()
            error[target & positive] = (0, 255, 0)     # TP
            error[~target & positive] = (0, 0, 255)    # FP
            error[target & ~positive] = (255, 0, 0)    # FN
            panels = [("Original", image), ("Ground truth", self.draw(image, target.astype(np.float32))),
                      ("Prediction", overlay), ("TP green / FP red / FN blue", error)]
        labelled = []
        for title, panel in panels:
            panel = cv2.copyMakeBorder(panel, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            cv2.putText(panel, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
            labelled.append(panel)
        return np.concatenate(labelled, axis=1)

    @staticmethod
    def _validate_probabilities(probabilities: np.ndarray, shape: tuple) -> None:
        if probabilities.ndim != 2 or probabilities.shape != shape:
            raise ValueError("概率图必须是与原图同尺寸的二维数组")
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("概率图包含 NaN/Inf 或超出 0~1;请先对 logits 使用 sigmoid")
