#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : reconnect.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : ReconnectingSource —— 断线重连(基于 end_reason 区分停/断; 正交增强, 不进注册表)
"""
断线重连包装器。给会断的源(网络流 / 摄像头)包一层自动重连。

工作原理(依赖 §10.2 的协议补丁):
    inner.read() 返回 None 时看 inner.end_reason —
        COMPLETED(或其它不可恢复原因) → 透传 None, 上层停
        DISCONNECTED / ERROR          → 指数退避后重开 inner, 成功则继续出帧

对上层透明: for frame in src 要么拿到帧, 要么只在真正结束时拿到 None。
对文件优雅降级: 文件读完报 COMPLETED, 本 wrapper 正常停(不会重开文件)。
不是一种源: 无 @register_source; 由工厂便捷函数或调用方手动套上。
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Optional

from ..core.base  import FrameSource
from ..core.types import Frame, SourceType, EndReason, is_recoverable


logger = logging.getLogger(__name__)


class ReconnectingSource(FrameSource):
    """断线重连包装器(适用于网络流 / 摄像头)。"""

    def __init__(
        self,
        inner: FrameSource,
        *,
        max_retries: Optional[int] = None,          # None = 无限重试
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
        backoff_factor: float = 2.0,
        on_reconnect: Optional[Callable[[int, EndReason], None]] = None,   # (第几次, 原因) 回调
    ):
        super().__init__(inner.config)              # 复用 inner 的 config
        self._inner = inner
        self._max_retries = max_retries
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._backoff_factor = backoff_factor
        self._on_reconnect = on_reconnect
        self._opened = False
        self._stop = threading.Event()              # 用于可打断的退避睡眠

    # ── 可被 close() 打断的睡眠 ──
    def _sleep(self, seconds: float) -> bool:
        """退避睡眠。返回 True 表示期间被要求停止(close 被调用)。"""
        return self._stop.wait(seconds)

    # ── FrameSource 协议 ──
    def open(self) -> bool:
        self._frame_index = 0
        self._stop.clear()
        self._end_reason = EndReason.RUNNING
        self._opened = self._open_with_retry()
        return self._opened

    def _open_with_retry(self) -> bool:
        """首次打开也带重试(冷启动时对端可能还没就绪)。"""
        attempt = 0
        backoff = self._backoff_initial
        while not self._stop.is_set():
            if self._inner.open():
                return True
            attempt += 1
            if self._max_retries is not None and attempt > self._max_retries:
                logger.error(f"首次打开失败, 已重试 {attempt - 1} 次仍不成, 放弃")
                self._end_reason = EndReason.DISCONNECTED
                return False
            logger.warning(f"打开失败, {backoff:.1f}s 后第 {attempt} 次重试: {self.source_path}")
            if self._sleep(backoff):
                return False                        # 睡眠中被 close
            backoff = min(backoff * self._backoff_factor, self._backoff_max)
        return False

    def read(self) -> Optional[Frame]:
        if not self._opened:
            self._end_reason = EndReason.NOT_STARTED
            return None

        frame = self._inner.read()
        if frame is not None:
            return frame                            # 正常出帧, 直接透传

        reason = self._inner.end_reason
        if not is_recoverable(reason):
            self._end_reason = reason               # 干净结束/不可恢复 → 透传, 上层停
            return None

        return self._reconnect_and_read(reason)     # 可恢复 → 退避重连

    def _reconnect_and_read(self, reason: EndReason) -> Optional[Frame]:
        attempt = 0
        backoff = self._backoff_initial
        while not self._stop.is_set():
            attempt += 1
            if self._max_retries is not None and attempt > self._max_retries:
                logger.error(f"重连放弃(中断原因 {reason.value}): 已重试 {attempt - 1} 次")
                self._end_reason = EndReason.DISCONNECTED
                return None

            logger.warning(f"源中断({reason.value}), {backoff:.1f}s 后第 {attempt} 次重连: {self.source_path}")
            if self._sleep(backoff):
                self._end_reason = EndReason.DISCONNECTED
                return None                         # 退避中被 close

            try:
                self._inner.close()                 # 先关掉旧句柄
            except Exception as e:
                logger.debug(f"重连前关闭旧句柄出错(忽略): {e}")

            if self._inner.open():
                if self._on_reconnect is not None:
                    try: self._on_reconnect(attempt, reason)
                    except Exception as e: logger.debug(f"on_reconnect 回调出错(忽略): {e}")
                logger.info(f"第 {attempt} 次重连成功: {self.source_path}")
                frame = self._inner.read()
                if frame is not None:
                    return frame                    # 连上且拿到帧
                # 刚连上又立刻空/断: 继续退避重试(不重置 backoff, 防抖动猛连)
            backoff = min(backoff * self._backoff_factor, self._backoff_max)
        # 被 close
        self._end_reason = EndReason.DISCONNECTED
        return None

    def close(self) -> None:
        self._opened = False
        self._stop.set()                            # 打断任何退避睡眠
        self._inner.close()

    # ── 透传 inner(重连不改变'是什么源、能产什么')──
    def get_source_type(self) -> SourceType:
        return self._inner.get_source_type()

    @property
    def modalities(self) -> frozenset[str]:
        return self._inner.modalities

    @property
    def seekable(self) -> bool:
        return False                                # 重连语义下 seek 无意义(会话会重置), 直接不支持

    def set_stride(self, stride: int) -> None:
        self._inner.set_stride(stride)

    @property
    def stride(self) -> int:
        return self._inner.stride
