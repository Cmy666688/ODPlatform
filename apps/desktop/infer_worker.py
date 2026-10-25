#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : infer_worker.py
# @Project   : ODPlatform
# @Function  : QThread 中执行推理,把事件转成 Qt 信号
"""每次推理一个 worker。取消直接调用线程安全的 CancelToken。

推理阻塞 worker 的事件循环,Stop 不能排队成 worker 槽调用,否则要等推理结束
才收到取消。token 在启动前创建,启动后立即 Stop 也不会被重新赋值吞掉。
"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from od_platform.inference import CancelToken, InferHooks, InferResult, OutputSink, ProgressEvent, infer_yolo

logger = logging.getLogger(__name__)


class InferWorker(QObject):
    """只运行服务和发送信号,所有 widget 操作留在主线程。"""
    progress_changed = Signal(int, int, float)
    inference_finished = Signal(object)          # 无论成功/失败,恰好一次 InferResult
    inference_failed = Signal(str)

    def __init__(self, model: str, source: str, sink: OutputSink,
                 cancel_token: CancelToken, backend: str = "yolo") -> None:
        super().__init__()
        self.model = model
        self.source = source
        self.sink = sink
        self.cancel_token = cancel_token
        self.backend = backend

    @Slot()
    def start_infer(self) -> None:
        logger.info(f"InferWorker.start_infer: model={self.model}, source={self.source}")
        hooks = InferHooks(on_progress=self._on_progress, progress_interval_frames=15)
        try:
            if self.backend == "mixercseg":
                from od_platform.backends.mixercseg.model import read_checkpoint, resolve_checkpoint
                from od_platform.inference.mixercseg_service import infer_mixercseg
                from od_platform.runtime_config.mixercseg import load_mixercseg_config
                state = read_checkpoint(resolve_checkpoint(self.model))
                bins = {int(v.shape[1]) for k, v in state["model"].items() if k.endswith("hog_feat.0.weight")}
                if len(bins) != 1:
                    raise ValueError("权重中 HOG bins 不一致")
                config = load_mixercseg_config("mixercseg_infer", {"model": self.model, "hog_bins": bins.pop()})
                result = infer_mixercseg(config, self.source, save=False, show=False,
                                         output_sink=self.sink, hooks=hooks, cancel_token=self.cancel_token)
            else:
                result = infer_yolo(
                    cli_args={"source": self.source, "model": self.model, "show": False, "save": False},
                    output_sink=self.sink, hooks=hooks, cancel_token=self.cancel_token,
                )
        except Exception as e:
            logger.exception("InferWorker 捕获到 service 异常")
            result = InferResult(success=False, output_dir=Path("unknown"), error=str(e))
        if not result.success:
            self.inference_failed.emit(result.error or "推理失败")
        self.inference_finished.emit(result)

    def _on_progress(self, evt: ProgressEvent) -> None:
        total = evt.total_frames if evt.total_frames is not None else -1
        self.progress_changed.emit(evt.frame_idx, total, evt.fps_loop)
