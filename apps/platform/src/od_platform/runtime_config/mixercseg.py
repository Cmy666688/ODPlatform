#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : mixercseg.py
# @Project   : ODPlatform
# @Function  : MixerCSeg 配置 —— 独立于 YOLO 参数表
"""默认训练参数来自 MixerCSeg main.py,不把语义分割损失塞进 YOLO 配置。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from od_platform.common import paths

UPSTREAM_COMMIT = "7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d"


class MixerCSegConfig(BaseModel):
    """论文代码的参数 + 平台运行开关。输入分辨率按原模型固定 512。"""
    model_config = ConfigDict(extra="forbid")
    backend: Literal["mixercseg"] = "mixercseg"
    data: str = "crackmap"
    model: Optional[str] = None                  # 训练默认从头;推理必须提供 checkpoint
    device: str = "auto"                        # 本机无 CUDA 时只改设备,不改训练配方
    BCELoss_ratio: float = Field(default=0.87, ge=0, le=1)
    DiceLoss_ratio: float = Field(default=0.13, ge=0, le=1)
    Norm_Type: Literal["GN"] = "GN"
    hog_bins: int = Field(default=36, ge=1)       # 源码=36;官方 CrackMap 权重=180,必须显式区分
    batch_size_train: int = Field(default=1, ge=1)
    batch_size_test: int = Field(default=1, ge=1)
    lr_scheduler: Literal["PolyLR"] = "PolyLR"
    lr: float = Field(default=5e-4, gt=0)
    min_lr: float = Field(default=1e-6, ge=0)
    weight_decay: float = Field(default=0.01, ge=0)
    epochs: int = Field(default=50, ge=1)
    seed: int = 42
    num_threads: int = Field(default=1, ge=0)      # 原始 DataLoader 参数
    load_width: Literal[512] = 512
    load_height: Literal[512] = 512
    torch_threads: int = Field(default=4, ge=1)
    experiment_name: str = "mixercseg-crackmap"
    threshold: float = Field(default=0.5, gt=0, lt=1)
    max_train_batches: Optional[int] = Field(default=None, ge=1)
    max_eval_batches: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_loss_and_lr(self) -> "MixerCSegConfig":
        if abs(self.BCELoss_ratio + self.DiceLoss_ratio - 1.0) > 1e-8:
            raise ValueError("BCELoss_ratio + DiceLoss_ratio 必须等于 1")
        if self.min_lr >= self.lr:
            raise ValueError("min_lr 必须小于 lr")
        return self


def load_mixercseg_config(reference: str = "mixercseg_train",
                          overrides: Optional[dict[str, Any]] = None) -> MixerCSegConfig:
    path = Path(reference)
    if not path.is_absolute() and "/" not in reference and "\\" not in reference:
        path = paths.RUNTIME_CONFIGS_DIR / (reference if reference.endswith(".yaml") else reference + ".yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"配置必须是映射: {path}")
    raw.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return MixerCSegConfig(**raw)
