#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : aio.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : AsyncSource —— 同步源 → 异步接口适配(不继承 FrameSource, 守 LSP)
"""
异步适配器: 把同步 FrameSource 适配成 async 接口, 供事件循环中的调用方 async for。

用 asyncio.to_thread 把阻塞的 open/read/close 丢到线程池, 不卡事件循环。

刻意不继承 FrameSource: 它的 read 是协程, 无法替换同步协议(LSP), 故是独立适配器,
提供 async for 而非 for。config/modalities/可 seek 性照常透传(与同步/异步无关)。
"""
from __future__ import annotations

import asyncio
from typing import Optional

from ..core.base   import FrameSource
from ..core.config import SourceConfig
from ..core.types  import Frame, SourceType


class AsyncSource:
    """把同步 FrameSource 适配成异步接口(不是 FrameSource 子类)。"""

    def __init__(self, inner: FrameSource):
        self._inner = inner

    # ── 透传描述性属性(与同步/异步无关)──
    @property
    def config(self) -> SourceConfig:
        return self._inner.config

    @property
    def source_path(self) -> str:
        return self._inner.source_path

    @property
    def modalities(self) -> frozenset[str]:
        return self._inner.modalities

    @property
    def seekable(self) -> bool:
        return self._inner.seekable

    @property
    def stride(self) -> int:
        return self._inner.stride

    def set_stride(self, stride: int) -> None:
        self._inner.set_stride(stride)

    def get_source_type(self) -> SourceType:
        return self._inner.get_source_type()

    # ── 异步方法(阻塞调用丢线程池)──
    async def open(self) -> bool:
        return await asyncio.to_thread(self._inner.open)

    async def read(self) -> Optional[Frame]:
        return await asyncio.to_thread(self._inner.read)

    async def close(self) -> None:
        await asyncio.to_thread(self._inner.close)

    async def seek(self, frame: Optional[int] = None, time_sec: Optional[float] = None) -> bool:
        return await asyncio.to_thread(self._inner.seek, frame, time_sec)

    # ── 异步上下文管理器 + 异步迭代器 ──
    async def __aenter__(self) -> "AsyncSource":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        await self.close()
        return False

    def __aiter__(self) -> "AsyncSource":
        return self

    async def __anext__(self) -> Frame:
        frame = await self.read()
        if frame is None:
            raise StopAsyncIteration
        return frame
