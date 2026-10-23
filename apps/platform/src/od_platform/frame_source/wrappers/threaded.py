#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : threaded.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : ThreadedSource —— 后台线程采集, 主循环取最新帧(正交增强, 不进注册表)
"""
线程化包装器: 把任意 FrameSource 的采集丢到后台线程, 主循环取帧不被采集阻塞。

正交性: 不认任何具体源, 只认 FrameSource 协议; 透传 inner 的类型与 modalities。
不是一种源: 没有 @register_source, 由工厂便捷函数或调用方手动套上。

缓冲策略:
    LATEST  —— 只留最新 1 帧(实时优先, 丢旧帧), 适合摄像头
    BOUNDED —— 有界队列(不丢帧, 满了阻塞采集), 适合离线视频并行解码
"""
from __future__ import annotations

import logging
import queue
import threading
from enum import Enum
from typing import Optional

from ..core.base  import FrameSource
from ..core.types import Frame, SourceType


logger = logging.getLogger(__name__)


class BufferStrategy(str, Enum):
    LATEST  = "latest"     # 只保留最新帧, 丢旧帧(实时优先)
    BOUNDED = "bounded"    # 有界队列, 不丢帧, 满则阻塞采集


class ThreadedSource(FrameSource):
    """后台线程采集包装器。"""

    def __init__(
        self,
        inner: FrameSource,
        *,
        strategy: BufferStrategy = BufferStrategy.LATEST,
        maxsize: int = 1,
        warmup: bool = True,
        read_timeout: float = 5.0,
    ):
        super().__init__(inner.config)      # 复用 inner 的 config → source_path 等有值
        self._inner   = inner
        self._strategy = strategy
        qsize = 1 if strategy == BufferStrategy.LATEST else max(1, maxsize)
        self._queue: "queue.Queue[Frame]" = queue.Queue(maxsize=qsize)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._eos  = threading.Event()      # inner 耗尽标志
        self._warmup = warmup
        self._read_timeout = read_timeout
        self._timeout_warned = False

    # ── 采集线程 ──
    def _capture_loop(self) -> None:
        while not self._stop.is_set():
            frame = self._inner.read()
            if frame is None:
                self._eos.set()
                break
            if self._strategy == BufferStrategy.LATEST:
                try: self._queue.get_nowait()       # 丢掉旧帧
                except queue.Empty: pass
                try: self._queue.put_nowait(frame)
                except queue.Full: pass
            else:  # BOUNDED: 满则阻塞等消费, 但可被 stop 打断
                while not self._stop.is_set():
                    try:
                        self._queue.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def _drain(self) -> None:
        while not self._queue.empty():
            try: self._queue.get_nowait()
            except queue.Empty: break

    # ── FrameSource 协议 ──
    def open(self) -> bool:
        self._frame_index = 0
        if not self._inner.open():
            return False
        self._stop.clear(); self._eos.clear(); self._timeout_warned = False
        self._drain()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

        if self._warmup:    # 等第一帧就位, 让 open 返回时已有帧可读
            for _ in range(int(self._read_timeout * 20)):
                if not self._queue.empty() or self._eos.is_set():
                    break
                threading.Event().wait(0.05)
        return True

    def read(self) -> Optional[Frame]:
        if self._thread is None:
            return None
        while not self._stop.is_set():
            try:
                return self._queue.get(timeout=self._read_timeout)
            except queue.Empty:
                if self._eos.is_set():
                    return None     # 采集线程已结束且队列取空 → 真结束
                if not self._timeout_warned:
                    logger.warning(
                        f"ThreadedSource: {self._read_timeout}s 未取到帧"
                        f"(采集慢或卡住), 后续不再重复此告警"
                    )
                    self._timeout_warned = True
                # 非 EOS: 继续等(实时源可能恢复)
        return None

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._inner.close()
        self._drain()

    # ── 透传 inner(线程化不改变'是什么源、能产什么')──
    def get_source_type(self) -> SourceType:
        return self._inner.get_source_type()

    @property
    def modalities(self) -> frozenset[str]:
        return self._inner.modalities

    @property
    def seekable(self) -> bool:
        return self._inner.seekable

    def seek(self, frame=None, time_sec=None) -> bool:
        # 注: 采集线程运行中 seek 有竞态, 建议仅在离线场景谨慎使用
        return self._inner.seek(frame=frame, time_sec=time_sec)

    def set_stride(self, stride: int) -> None:
        self._inner.set_stride(stride)      # 转发给 inner, 由它用最快方式跳采

    @property
    def stride(self) -> int:
        return self._inner.stride
