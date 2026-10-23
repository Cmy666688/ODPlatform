#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : frame_source 包入口 —— 公共 API + 触发内置源自注册
"""
frame_source: 统一帧输入源抽象层(可插拔 · 多模态)。

把任意输入源封装成同一套 `for frame in source` 接口。派发用配置类型(注册表),
加源用一行 @register_source(内置或第三方, 不改本模块), Frame 支持多模态副平面。

最常用:
    from od_platform.frame_source import create_frame_source
    with create_frame_source("0") as src:          # "0"/"x.mp4"/"rtsp://..."/"./imgs"
        for frame in src:
            model(frame.image)

显式选型(硬件/内存源):
    from od_platform.frame_source import create_frame_source, CameraConfig
    with create_frame_source(CameraConfig(fps=90, backend="msmf")) as src:
        ...

第三方源发现(可选): 见 load_plugins()。
"""
from __future__ import annotations

__version__ = "0.2.0"     # 0.1=首发; 0.2=上线加固(EndReason + 断线重连)

# ── core: 类型 / 配置 / 协议 / 注册表 ──
from .core import (
    SourceType, FrameInfo, Frame, CameraIntrinsics, EndReason, is_recoverable,
    IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, STREAM_SCHEMES, is_stream_url,
    SourceConfig, CameraConfig, VideoConfig, ImageConfig, ImageFolderConfig,
    CameraBackend, CameraCodec,
    FrameSource,
    register_source, build_from_config, build_from_string, list_sources,
)

# ── sources: import 即触发四个内置源自注册(这几行是注册的唯一触发点, 勿删)──
from .sources import CameraSource, VideoSource, ImageSource, ImageFolderSource

# ── wrappers ──
from .wrappers import ThreadedSource, BufferStrategy, AsyncSource, ReconnectingSource

# ── factory: 主入口 ──
from .factory import create_frame_source, create_threaded_source, create_async_source

# ── 可选: 第三方源发现(默认不自动调用, 交给使用者)──
from .plugins import load_plugins

__all__ = [
    "__version__",
    # 类型
    "SourceType", "FrameInfo", "Frame", "CameraIntrinsics", "EndReason", "is_recoverable",
    "IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS", "STREAM_SCHEMES", "is_stream_url",
    # 配置
    "SourceConfig", "CameraConfig", "VideoConfig", "ImageConfig", "ImageFolderConfig",
    "CameraBackend", "CameraCodec",
    # 协议 + 注册表
    "FrameSource", "register_source", "build_from_config", "build_from_string", "list_sources",
    # 内置源
    "CameraSource", "VideoSource", "ImageSource", "ImageFolderSource",
    # wrappers
    "ThreadedSource", "BufferStrategy", "AsyncSource", "ReconnectingSource",
    # 工厂
    "create_frame_source", "create_threaded_source", "create_async_source",
    # 插件
    "load_plugins",
]
