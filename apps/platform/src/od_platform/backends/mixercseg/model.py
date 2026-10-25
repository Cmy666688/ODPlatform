#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : model.py
# @Project   : ODPlatform
# @Function  : 原版 MixerCSeg 构造、设备选择和安全权重读取
"""只适配依赖/设备,不替换网络结构。官方 CrackMap 权重可 strict=True 加载。"""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

from od_platform.common import paths
from od_platform.runtime_config.mixercseg import MixerCSegConfig


def resolve_device(requested: str) -> str:
    import torch
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested
    if device.isdigit():
        device = "cuda:" + device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA,但当前 PyTorch 没有可用 CUDA。可显式传 --device cpu")
    return device


def resolve_checkpoint(reference: str) -> Path:
    direct = Path(reference)
    for candidate in (direct, paths.ROOT_DIR / direct, paths.TRAINED_MODELS_DIR / direct,
                      paths.PRETRAINED_MODELS_DIR / direct):
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"MixerCSeg 权重不存在: {reference}")


def read_checkpoint(path: Path) -> dict[str, Any]:
    import torch
    # 官方 checkpoint 的 args 是 argparse.Namespace;只允许这个简单容器,不关闭安全反序列化。
    with torch.serialization.safe_globals([Namespace]):
        result = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(result, dict) or "model" not in result:
        raise ValueError(f"不是 MixerCSeg checkpoint(model 字段缺失): {path}")
    return result


def build_model(config: MixerCSegConfig, *, checkpoint: str | None = None):
    import torch
    try:
        from ._vendor.models.segmentor import build_MixerCSeg
    except ImportError as e:
        raise RuntimeError('缺少 MixerCSeg 依赖,请安装 pip install -e "./apps/platform[mixercseg]"') from e
    device = resolve_device(config.device)
    torch.set_num_threads(config.torch_threads)
    args = Namespace(**config.model_dump())
    args.device = device
    model, criterion = build_MixerCSeg(args)
    if config.hog_bins != 36:
        for module in model.modules():
            if module.__class__.__name__ == "HoGEdgeGateConv":
                old = module.hog_feat[0]
                module.nbins = config.hog_bins
                module.hog_feat[0] = torch.nn.Conv2d(config.hog_bins, old.out_channels, 1).to(device)
    if checkpoint is not None:
        state = read_checkpoint(resolve_checkpoint(checkpoint))
        bins = {int(v.shape[1]) for k, v in state["model"].items() if k.endswith("hog_feat.0.weight")}
        if bins != {config.hog_bins}:
            raise ValueError(f"checkpoint HOG bins={sorted(bins)},配置 hog_bins={config.hog_bins};请使用匹配的模型配置")
        model.load_state_dict(state["model"], strict=True)
    return model, criterion, device
