#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : service.py
# @Project   : ODPlatform
# @Function  : 训练编排 —— 顺序 + 验章 + 记录 + 归档
"""训练编排器。只做四件事: 顺序、验章(确认数据身份)、记录、归档。其余全调邻居:
    refs / RunContext(D3)、naming.run_stem(阶段3)、reconcile_data_identity(D4 identity)、
    log_config_report(D5)、TrainMetrics(阶段8)、ultralytics。
判据(阶段0.2 四问)在 _write_audit 兑现; 第4问(数据身份)由 identity 提供、D6 盖 train 章收尾。"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from od_platform.common import refs, stamp
from od_platform.common.lineage import TOOL_VERSION
from od_platform.common.paths import PRETRAINED_MODELS_DIR
from od_platform.common.run_context import RunContext
from od_platform.data_validation.identity import DataIdentity, reconcile_data_identity
from od_platform.runtime_config.report import log_config_report
from od_platform.training.archive import archive_best_weight
from od_platform.training.result import TrainMetrics, log_train_metrics

logger = logging.getLogger(__name__)

TRAINER_VERSION = "odp-train/0.4"      # D6 盖 train 章用


@dataclass(frozen=True)
class TrainResult:
    """一次训练的结果(纯数据)。★ 对外面板 —— cli/ 只认这几个字段。"""
    success:     bool
    run_id:      str
    save_dir:    Optional[Path] = None
    best_weight: Optional[Path] = None
    metrics:     Optional[TrainMetrics] = None
    identity:    Optional[DataIdentity] = None
    message:     str = ""


def _resolve_model_arg(model: str) -> str:
    """模型引用 -> 给 ultralytics 的字符串。本地有就用绝对路径, 没有原样交它自动下载。"""
    p = refs.resolve_ref(model, base_dir=PRETRAINED_MODELS_DIR, default_suffix=".pt")
    if p.exists():
        logger.info("模型(解析): %s", p)
        return str(p)
    logger.info("本地无 %s, 交给 ultralytics 自动下载: %s", p.name, model)
    return model


def _run_training(model_arg: str, config: Any, data_yaml: Path, run: RunContext) -> Any:
    """真正开训。名字在这之前已定死, save_dir 已知。"""
    from ultralytics import YOLO      # 延迟 import: 不训练的路径不吃这个重依赖
    model = YOLO(model_arg)
    kwargs = config.to_ultralytics_kwargs()
    kwargs.pop("data", None)
    kwargs.pop("model", None)
    kwargs.update(project=str(run.run_dir.parent), name=run.run_id, exist_ok=True)
    return model.train(data=str(data_yaml), **kwargs)


def _write_audit(run: RunContext, config: Any, merger: Optional[Any], data_yaml: Path,
                 stem: str, metrics: TrainMetrics, identity: Optional[DataIdentity]) -> Path:
    """把"这次怎么跑的"落进现场。★ 四问答卷:
        artifacts -> ① 产物一套(run_id/stem)   config/config_sources -> ② 参数+谁给的
        metrics   -> ③ 结果(run_id 必须自洽)     data_identity -> ④ 数据是哪一份/验没验"""
    audit = {
        "run_id": run.run_id, "created_at": run.created_at,     # ★ RunContext 那唯一一次 now()
        "artifacts": {"run_dir": str(run.run_dir), "stem": stem},
        "experiment_name": getattr(config, "experiment_name", None),
        "data_yaml": str(data_yaml),
        "data_identity": identity.to_dict() if identity is not None else None,   # ← 第4问
        "config": config.to_audit_snapshot(),
        "metrics": metrics.to_dict(),
    }
    if merger is not None:
        audit["config_sources"] = merger.to_audit_log()
    path = run.artifact_path("odp_audit.json")
    path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("审计快照已落盘: %s", path)
    return path


def train_yolo(*, config: Any, data_yaml: Path, run: RunContext, stem: str,
               merger: Optional[Any] = None, archive: bool = True,
               require_validated: bool = True) -> TrainResult:
    """训练一个 YOLO 模型。价值 = 顺序 + 验章 + 记录 + 归档:
        1) 记录参数从哪来   -> runtime_config.report
        2) ★ 验章: 确认数据身份 -> data_validation.identity(D6 自己算指纹, 与 D3/D4 章对账)
        3) 解析模型引用     -> refs
        4) 训练             -> ultralytics(名字训练前已定)
        5) 抽指标 + 记日志  -> TrainMetrics
        6) ★ 盖 train 章 + 落审计
        7) 归档             -> archive"""
    logger.info("=" * 60)
    logger.info("训练开始 | run_id=%s", run.run_id)
    logger.info("现场目录: %s | 产物 stem: %s", run.run_dir, stem)
    logger.info("实验名: %s | 数据集: %s", getattr(config, "experiment_name", None) or "(未命名)", data_yaml)
    logger.info("=" * 60)

    # 1) 参数从哪来(裂缝 B)
    if merger is not None:
        log_config_report(config, merger, logger=logger)

    # 2) ★ 验章(裂缝 D): D6 自己算 data_fingerprint, 与 D3/D4 的章三方对账。
    #    这是"确认"不是"发明"(阶段 4.5): 算出来是为了核对"我摸到的是不是上游说的那个"。
    identity = reconcile_data_identity(data_yaml, require_validated=require_validated)
    if not identity.ok:
        logger.error("数据身份核验未通过, 拒绝开训: %s", identity.reason)
        return TrainResult(success=False, run_id=run.run_id, identity=identity,
                           message=f"数据身份核验未通过: {identity.reason}")

    # 3) 模型引用 -> 路径
    model_ref = getattr(config, "model", None) or "yolov8n.pt"
    logger.info("模型(声明): %s", model_ref)
    model_arg = _resolve_model_arg(model_ref)

    # 4) 训练
    try:
        results = _run_training(model_arg, config, data_yaml, run)
    except KeyboardInterrupt:
        logger.warning("训练被手动中断 (Ctrl-C) | run_id=%s | 现场保留: %s", run.run_id, run.run_dir)
        return TrainResult(success=False, run_id=run.run_id, save_dir=run.run_dir,
                           identity=identity, message="训练被手动中断")
    except Exception as e:
        logger.exception("训练失败: %s", e)
        return TrainResult(success=False, run_id=run.run_id, save_dir=run.run_dir,
                           identity=identity, message=f"训练异常: {e}")

    # 5) 指标(run_id 注入 —— 不许自己拍表)
    metrics = TrainMetrics.from_yolo_results(results, run_id=run.run_id)
    log_train_metrics(metrics, logger=logger)

    # 6) ★ 盖 train 章: 我(D6)训练吃了这批数据。data_fingerprint 用 identity 里
    #    D6 自己算的那个(一手, 阶段 4.5), 不抄 D3/D4。
    if identity.data_fingerprint:
        stamp.write_stamp(
            stage_file=stamp.STAMP_TRAIN, data_fingerprint=identity.data_fingerprint,
            run_id=run.run_id, tool_version=TRAINER_VERSION,
            payload={"stem": stem, "experiment_name": getattr(config, "experiment_name", None),
                     "reconciled": identity.reconciled},
        )
    _write_audit(run, config, merger, data_yaml, stem, metrics, identity)

    # 7) 归档(裂缝 C), 名字与日志同源
    best = archive_best_weight(run.run_dir, stem) if archive else None
    logger.info("训练完成 | run_id=%s | 现场: %s", run.run_id, run.run_dir)
    return TrainResult(success=True, run_id=run.run_id, save_dir=run.run_dir,
                       best_weight=best, metrics=metrics, identity=identity)
