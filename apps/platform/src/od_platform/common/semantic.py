#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : semantic.py
# @Project   : ODPlatform
# @Function  : RGB 图像 / 二值掩膜与语义数据集契约
"""语义分割的公共数据语言。alpha 不是类别,指纹同时包含图像和掩膜内容。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from od_platform.common import paths


def read_pixels(path: Path) -> np.ndarray:
    try:
        array = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as e:
        raise ValueError(f"无法解码图像: {path}") from e
    if array is None:
        raise ValueError(f"无法解码图像: {path}")
    return array


def read_bgr(path: Path) -> np.ndarray:
    image = read_pixels(path)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] in (3, 4):
        color_image: np.ndarray = image[..., :3].copy()  # 显式去 alpha,不把透明度当输入通道
        return color_image
    raise ValueError(f"不支持的图像通道: {path}: {image.shape}")


def read_binary_mask(path: Path) -> np.ndarray:
    mask = read_pixels(path)
    if mask.ndim == 3:
        color = mask[..., :3]                   # alpha 先去除,再判断 RGB 是否等价
        if not np.all(color == color[..., :1]):
            raise ValueError(f"彩色掩膜没有声明类别映射: {path}")
        mask = color[..., 0]
    if mask.ndim != 2 or not set(np.unique(mask).tolist()) <= {0, 1, 255}:
        raise ValueError(f"掩膜不是 0/1 或 0/255 二值标注: {path}")
    return (mask > 0).astype(np.uint8) * 255


def write_png(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", pixels)
    if not ok:
        raise OSError(f"图像编码失败: {path}")
    encoded.tofile(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def semantic_fingerprint(samples: list[dict[str, Any]]) -> str:
    rows = sorted((s["split"], s["stem"], s["image_sha256"], s["mask_sha256"]) for s in samples)
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def load_semantic_dataset(yaml_path: Path) -> tuple[dict[str, Any], Path]:
    document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("format") != "semantic_masks":
        raise ValueError(f"不是语义掩膜数据集配置: {yaml_path}")
    root = Path(document["path"])
    if not root.is_absolute():
        root = paths.ROOT_DIR / root
    return document, root


def paired_samples(root: Path, split: str) -> list[tuple[Path, Path]]:
    images = sorted((root / "images" / split).glob("*.png"))
    return [(p, root / "masks" / split / p.name) for p in images]
