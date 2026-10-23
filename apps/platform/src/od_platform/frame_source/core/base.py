#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : base.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : FrameSource 抽象基类 — 统一所有输入源的协议(config 驱动 + 能力发现)
"""
帧源抽象基类。定义所有输入源(摄像头/视频/图片/包装器/第三方源)共享的协议:

    __init__(config)               config 驱动: 源吃 SourceConfig, 不吃裸字符串
    open / read / close            三个抽象方法, 子类必须实现
    get_source_type()              返回源类型(内置枚举或第三方字符串)
    modalities                     能力发现: 本源能产哪些平面, 默认 {'image'}
      + seek                       一次性跳转, 子类按能力覆盖
      + set_stride                 持续跳采, 子类按能力在 read() 内实现
      + 上下文管理器
      + 迭代器

帧位置控制原语(同族, 协议层):
    seek(frame=N)      一次性跳到第 N 帧
    seek(time_sec=T)   一次性跳到 T 秒
    set_stride(N)      之后每 N 帧产出 1 帧, 直到再次调用

为什么 stride 是协议方法而不是 wrapper:
    1. 语义同族 — 跟 seek 一样是"源对自身行为的控制原语"
    2. 性能要求 — 每个源能用自己最快的跳帧手段:
       VideoSource 用 cap.grab() 跳过 YUV→BGR 转换;
       ImageFolderSource 直接跳索引, 零 IO;
       wrapper 路径只能 read-and-discard, 浪费完整解码。
    3. 生命周期 — set_stride() 可在迭代过程中动态切换。

子类实现 stride 的模板(在 read() 开头加几行):
    def read(self):
        # stride>1 且非首帧: 用本源最快的方式跳过 stride-1 帧
        if self._stride > 1 and self._frame_index > 0:
            if not self._fast_skip(self._stride - 1):
                return None
        ...原有读帧逻辑...
    其中 _fast_skip 是私有方法, 各源自己写, 不入协议。
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Iterator, Optional

from .config import SourceConfig
from .types  import Frame, SourceType, EndReason


logger = logging.getLogger(__name__)


class FrameSource(ABC):
    """
    帧源抽象基类(config 驱动)。

    标准用法:
        with SomeSource(some_config) as source:
            for frame in source:
                process(frame.image)
                print(frame.width, frame.height)

    能力发现(开循环前决策):
        if "depth" in source.modalities:
            prepare_pointcloud_pipeline()

    跳帧(seek, 一次性):
        source.seek(frame=100)      # 跳到第 100 帧
        source.seek(time_sec=3.5)   # 跳到 3.5 秒

    跳采(stride, 持续):
        source.set_stride(3)        # 之后每 3 帧产出 1 帧
        source.set_stride(1)        # 恢复逐帧

    注意:
        同一 source 对象只能迭代一次(标准迭代器协议)。
        需要重头迭代请重新 open() 或重新创建对象。
    """

    def __init__(self, config: SourceConfig):
        self.config       = config
        self.source_path  = config.identity     # 日志/调试用的人类可读标识
        self._frame_index = 0
        self._stride      = 1     # 默认透传; >1 时每 stride 帧产出 1 帧
        self._start_time  = time.time()
        self._end_reason: EndReason = EndReason.NOT_STARTED   # 上次迭代因何停止(read() 返回 None 后查询)

    # ── 子类必须实现 ────────────────────────────────────────
    @abstractmethod
    def open(self) -> bool:
        """打开输入源,返回是否成功。实现必须复位位置状态(见类 docstring 契约)。"""

    @abstractmethod
    def read(self) -> Optional[Frame]:
        """
        读取一帧,源耗尽时返回 None。

        实现 stride 的源(VideoSource / ImageFolderSource)在本方法开头
        做"跳过 stride-1 帧"动作; 不实现 stride 的源(CameraSource /
        ImageSource)忽略 self._stride 即可。
        """

    @abstractmethod
    def close(self) -> None:
        """关闭输入源,释放资源。"""

    @abstractmethod
    def get_source_type(self) -> SourceType:
        """返回输入源类型(内置源返回 SourceType 成员; 第三方源可返回自己的字符串)。"""

    # ── 能力发现(默认只产主图, 多模态源覆盖)────────────────
    @property
    def end_reason(self) -> EndReason:
        """
        上一次迭代因何停止。在 read() 返回 None 之后查询, 用于区分:
            COMPLETED             —— 干净读完, 上层该停
            DISCONNECTED / ERROR  —— 异常中断, 上层可重连(见 is_recoverable)
        read() 的签名不变(for frame in src 老循环零改), 这是旁挂的可查询状态。
        """
        return self._end_reason

    @property
    def modalities(self) -> frozenset[str]:
        """
        本源能产出的平面集合。默认 {'image'}(只有主视觉平面)。
        深度源覆盖为 {'image', 'depth'}, 双目源覆盖为 {'image', 'right'} 等。

        供调用方在开循环前决策是否准备额外管线。
        """
        return frozenset({"image"})

    # ── seek 协议(默认不支持,子类按需覆盖)─────────────────
    def seek(
        self,
        frame: Optional[int] = None,
        time_sec: Optional[float] = None,   # 参数名 time_sec 避免与 time 模块冲突
    ) -> bool:
        """跳转到指定位置(基类默认不支持)。返回是否成功。"""
        logger.warning(f"{self.__class__.__name__} 不支持 seek 操作")
        return False

    @property
    def seekable(self) -> bool:
        """是否支持 seek,子类覆盖返回 True。"""
        return False

    # ── stride 协议(持续跳采)──────────────────────────────
    def set_stride(self, stride: int) -> None:
        """
        设置跳采步长。stride=1 透传(默认), stride=N 每 N 帧产出 1 帧。

        基类只登记意图(self._stride), 实际跳帧由子类在 read() 内实现。
        不支持/不适用跳帧的源(摄像头、单图)应覆盖本方法, 给出警告或忽略。

        Raises:
            ValueError: stride < 1
        """
        if stride < 1:
            raise ValueError(f"stride 必须 ≥ 1, 收到: {stride}")
        self._stride = stride
        if stride > 1:
            logger.debug(f"{self.__class__.__name__}: stride 设为 {stride}")

    @property
    def stride(self) -> int:
        """当前跳采步长(1 表示透传)。"""
        return self._stride

    # ── 上下文管理器协议 ────────────────────────────────────
    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False  # 不吞异常

    # ── 迭代器协议 ──────────────────────────────────────────
    def __iter__(self) -> Iterator[Frame]:
        return self

    def __next__(self) -> Frame:
        frame = self.read()
        if frame is None:
            raise StopIteration
        return frame
