"""语义像素指标。固定概率阈值,不做每张图的 max 归一化。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BinaryMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, probabilities: np.ndarray, truth: np.ndarray, threshold=0.5) -> None:
        prediction, target = probabilities >= threshold, truth > 0.5
        if prediction.shape != target.shape or not np.isfinite(probabilities).all():
            raise ValueError("像素指标输入尺寸不一致或预测存在非有限值")
        self.tp += int(np.count_nonzero(prediction & target))
        self.fp += int(np.count_nonzero(prediction & ~target))
        self.fn += int(np.count_nonzero(~prediction & target))
        self.tn += int(np.count_nonzero(~prediction & ~target))

    def to_dict(self) -> dict[str, float]:
        fg_union = self.tp + self.fp + self.fn
        bg_union = self.tn + self.fp + self.fn
        foreground = self.tp / fg_union if fg_union else 1.0
        background = self.tn / bg_union if bg_union else 1.0
        f1_denom = 2 * self.tp + self.fp + self.fn
        return {"mIoU": (foreground + background) / 2, "foreground_IoU": foreground,
                "F1": 2 * self.tp / f1_denom if f1_denom else 1.0,
                "precision": self.tp / (self.tp + self.fp) if self.tp + self.fp else float(self.fn == 0),
                "recall": self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0}
