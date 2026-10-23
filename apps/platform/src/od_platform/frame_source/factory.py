#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : factory.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 便捷工厂 —— str|config 收口到注册表, 三个入口(普通/线程/异步)
"""
便捷工厂: frame_source 对外的主入口。

薄: 真正的"认哪种源"逻辑在 registry 和各源自己手里, 工厂只做一个判断——
输入是 SourceConfig 还是 str, 分别交给 build_from_config / build_from_string。

不 import 任何具体源(依赖倒置): 源靠"被 import sources 时自注册"进入注册表,
所以加一种源工厂零改动(规矩 E)。
"""
from __future__ import annotations

from .core.config   import SourceConfig
from .core.registry import build_from_config, build_from_string
from .core.base     import FrameSource
from .wrappers      import ThreadedSource, AsyncSource, BufferStrategy


def _resolve(source: str | SourceConfig) -> FrameSource:
    """把输入(字符串糖 或 显式 config)解析成一个尚未 open 的源。"""
    if isinstance(source, SourceConfig):
        return build_from_config(source)
    if isinstance(source, str):
        return build_from_string(source)
    raise TypeError(
        f"source 必须是 str 或 SourceConfig, 收到 {type(source).__name__}。"
        f"内存/硬件源请传对应的 config 对象。"
    )


def create_frame_source(source: str | SourceConfig, *, stride: int = 1) -> FrameSource:
    """
    创建一个帧源(最常用入口)。

    Args:
        source: 字符串糖("0"/"x.mp4"/"rtsp://..."/"./imgs")或显式 config
                (CameraConfig(...) / 未来的 DepthConfig(...) / 第三方 XxxConfig(...))。
        stride: 跳采步长(≥1); 不支持跳采的源(摄像头)会告警并忽略。

    Returns:
        尚未 open 的 FrameSource; 建议用 with 使用。
    """
    if stride < 1:
        raise ValueError(f"stride 必须 ≥ 1, 收到: {stride}")
    src = _resolve(source)
    if stride > 1:
        src.set_stride(stride)
    return src


def create_threaded_source(
    source: str | SourceConfig,
    *,
    stride: int = 1,
    strategy: BufferStrategy = BufferStrategy.LATEST,
    maxsize: int = 1,
    warmup: bool = True,
) -> ThreadedSource:
    """创建后台线程采集的源(实时推理防止采集被拖慢, 见 §5.1)。"""
    if stride < 1:
        raise ValueError(f"stride 必须 ≥ 1, 收到: {stride}")
    inner = _resolve(source)
    if stride > 1:
        inner.set_stride(stride)
    return ThreadedSource(inner, strategy=strategy, maxsize=maxsize, warmup=warmup)


def create_async_source(
    source: str | SourceConfig,
    *,
    stride: int = 1,
    threaded: bool = False,
    strategy: BufferStrategy = BufferStrategy.LATEST,
    maxsize: int = 1,
) -> AsyncSource:
    """创建异步接口的源(供 async for 使用, 见 §5.2)。threaded=True 时内部再套一层线程采集。"""
    if stride < 1:
        raise ValueError(f"stride 必须 ≥ 1, 收到: {stride}")
    inner = _resolve(source)
    if stride > 1:
        inner.set_stride(stride)
    if threaded:
        inner = ThreadedSource(inner, strategy=strategy, maxsize=maxsize)
    return AsyncSource(inner)
