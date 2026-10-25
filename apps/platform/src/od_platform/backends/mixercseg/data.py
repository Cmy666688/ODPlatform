#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : data.py
# @Project   : ODPlatform
# @Function  : 原版预处理 —— RGB, cubic resize, normalize(0.5),二值阈值127
"""掩膜在磁盘保留原始尺寸,训练时采用上游明确的插值/阈值规则。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from od_platform.common.semantic import paired_samples, read_bgr, read_binary_mask


def image_tensor(image: np.ndarray, width: int = 512, height: int = 512):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_CUBIC)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div(255).sub(0.5).div(0.5)


class CrackMaskDataset(Dataset):
    def __init__(self, root: Path, split: str, width=512, height=512):
        self.samples = paired_samples(root, split)
        self.width, self.height = width, height
        if not self.samples:
            raise ValueError(f"数据划分为空: {root}/{split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = image_tensor(read_bgr(image_path), self.width, self.height)
        mask = cv2.resize(read_binary_mask(mask_path), (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        label = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return {"image": image, "label": label, "image_path": str(image_path), "mask_path": str(mask_path)}
