#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : mixercseg_service.py
# @Project   : ODPlatform
# @Function  : 原版 MixerCSeg 训练配方 + 数据门禁 + 现场审计
"""AdamW + mmengine PolyLR + 原模型 BCE/Dice。

用 val 选 best,测试集不参与逐 epoch 模型选择;这是对上游测试集选模的显式修正。
所有缩短训练的开关都写进审计,不能把 smoke 结果当作完整训练。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
import random
import shutil
import time
from typing import Any

import numpy as np

from od_platform.common import paths
from od_platform.common.run_context import RunContext
from od_platform.common.semantic import load_semantic_dataset
from od_platform.data_validation.semantic import validate_semantic_dataset
from od_platform.runtime_config.mixercseg import MixerCSegConfig, UPSTREAM_COMMIT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MixerTrainResult:
    success: bool
    run_id: str
    save_dir: Path
    best_weight: Path | None = None
    message: str = ""


def evaluate_model(model, criterion, loader, device: str, threshold: float, max_batches=None) -> dict:
    import torch
    from od_platform.backends.mixercseg.metrics import BinaryMetrics
    model.eval()
    metrics = BinaryMetrics()
    losses = []; samples = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            inputs, target = batch["image"].to(device), batch["label"].to(device)
            logits = model(inputs)
            loss = criterion(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("验证损失不是有限数")
            losses.append(float(loss.item()))
            metrics.update(logits.sigmoid().cpu().numpy(), target.cpu().numpy(), threshold)
            samples += len(inputs)
    if not losses:
        raise RuntimeError("验证没有读取任何样本")
    return {**metrics.to_dict(), "loss": float(np.mean(losses)), "samples": samples}


def train_mixercseg(config: MixerCSegConfig, data_yaml: Path, run: RunContext,
                    *, archive=True) -> MixerTrainResult:
    import torch
    from torch.utils.data import DataLoader
    from od_platform.backends.mixercseg.data import CrackMaskDataset
    from od_platform.backends.mixercseg.model import build_model

    audit_path = run.artifact_path("odp_audit.json")
    audit: dict[str, Any] = {"backend": "mixercseg", "task": "semantic_segment", "run_id": run.run_id,
             "upstream_commit": UPSTREAM_COMMIT, "config": config.model_dump(),
             "smoke": config.max_train_batches is not None or config.max_eval_batches is not None,
             "model_selection_split": "val", "history": [], "success": False}
    try:
        report = validate_semantic_dataset(data_yaml)
        audit["validation"] = report.to_dict()
        if report.overall_severity == "ERROR":
            raise ValueError("数据质检 ERROR,拒绝开始训练")
        if report.overall_severity == "WARNING":
            logger.warning("数据存在风险提示,详见质检报告;原始划分保持不变")
        _, root = load_semantic_dataset(data_yaml)
        from mmengine.optim.scheduler.lr_scheduler import PolyLR
        torch.manual_seed(config.seed); np.random.seed(config.seed); random.seed(config.seed)
        model, criterion, device = build_model(config, checkpoint=config.model)
        audit["resolved_device"] = device
        audit["parameters"] = sum(p.numel() for p in model.parameters())
        train_set = CrackMaskDataset(root, "train", config.load_width, config.load_height)
        val_set = CrackMaskDataset(root, "val", config.load_width, config.load_height)
        generator = torch.Generator().manual_seed(config.seed)
        train_loader = DataLoader(train_set, batch_size=config.batch_size_train, shuffle=True,
                                  num_workers=config.num_threads, generator=generator)
        val_loader = DataLoader(val_set, batch_size=config.batch_size_test, shuffle=False,
                                num_workers=config.num_threads)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        scheduler = PolyLR(optimizer, eta_min=config.min_lr, begin=0, end=config.epochs)
        best_score = -1.0
        weights = run.artifact_path("weights", "last.pth")
        best_path = weights.with_name("best.pth")
        for epoch in range(config.epochs):
            model.train()
            losses = []; samples = 0; start = time.perf_counter()
            learning_rate = float(optimizer.param_groups[0]["lr"])
            for index, batch in enumerate(train_loader):
                if config.max_train_batches is not None and index >= config.max_train_batches:
                    break
                inputs, target = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), target)
                if not torch.isfinite(loss):
                    raise RuntimeError("训练损失不是有限数")
                loss.backward(); optimizer.step()
                losses.append(float(loss.item())); samples += len(inputs)
            if not losses:
                raise RuntimeError("训练没有读取任何样本")
            scheduler.step()
            validation = evaluate_model(model, criterion, val_loader, device, config.threshold,
                                        config.max_eval_batches)
            record = {"epoch": epoch + 1, "lr": learning_rate, "train_loss": float(np.mean(losses)),
                      "train_samples": samples, "validation": validation,
                      "seconds": round(time.perf_counter() - start, 3)}
            audit["history"].append(record)
            logger.info("MixerCSeg epoch=%s/%s loss=%.6f val_mIoU=%.4f lr=%.8f",
                        epoch + 1, config.epochs, record["train_loss"], validation["mIoU"], learning_rate)
            checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "lr_scheduler": scheduler.state_dict(), "epoch": epoch,
                          "config": config.model_dump(), "data_fingerprint": report.snapshot.data_fingerprint}
            torch.save(checkpoint, weights)
            if validation["mIoU"] > best_score:
                best_score = validation["mIoU"]; shutil.copy2(weights, best_path)
            audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        destination = best_path
        if archive:
            paths.TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            prefix = "mixercseg-smoke" if audit["smoke"] else "mixercseg"
            destination = paths.TRAINED_MODELS_DIR / f"{prefix}-{run.run_id}-{data_yaml.stem}-best.pth"
            shutil.copy2(best_path, destination)
        audit["success"] = True
        audit["best_weight"] = str(destination)
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        return MixerTrainResult(True, run.run_id, run.run_dir, destination)
    except (Exception, KeyboardInterrupt) as e:
        logger.exception("MixerCSeg 训练失败或被中断: %s", e)
        audit["error"] = str(e) or type(e).__name__
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
        return MixerTrainResult(False, run.run_id, run.run_dir, message=audit["error"])
