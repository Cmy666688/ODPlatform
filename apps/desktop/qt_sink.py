#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : qt_sink.py
# @Project   : ODPlatform
# @Function  : OutputSink → Qt 信号桥
"""把 renderer 线程的标注帧交给 Qt 主线程,不在引擎层引入 Qt。

QObject 与 OutputSink(ABC) 分别持有,避开两套 metaclass 的多继承冲突。
跨线程传独立、连续的图像副本,避免渲染端复用 buffer 后画面变化。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from od_platform.frame_source import SourceType
from od_platform.inference import OutputSink

logger = logging.getLogger(__name__)


class FrameSignals(QObject):
    """在 UI 线程构造,信号通过 QueuedConnection 送回 UI。"""
    frame_ready = Signal(object)             # annotated: BGR uint8 ndarray


class QtSignalSink(OutputSink):
    """OutputSink 的桌面实现。signals 的生命期随 sink,每次推理新建一份。"""

    def __init__(self) -> None:
        self.signals = FrameSignals()
        self.write_count = 0
        self._closed = True

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        self.write_count = 0
        self._closed = False
        logger.info(f"QtSignalSink.open: dir={output_dir}, source_type={source_type}")

    def write(self, frame, annotated: np.ndarray) -> None:
        if self._closed:
            return
        try:
            if annotated.dtype != np.uint8 or annotated.ndim != 3 or annotated.shape[2] != 3:
                raise ValueError("桌面显示要求 BGR uint8 三通道图像")
            self.signals.frame_ready.emit(annotated.copy(order="C"))
            self.write_count += 1
        except Exception as e:
            logger.warning(f"QtSignalSink.write 失败 (已吞): {e}")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            logger.info(f"QtSignalSink.close: 共发了 {self.write_count} 帧")
