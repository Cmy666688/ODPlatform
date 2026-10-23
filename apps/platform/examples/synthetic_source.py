#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : synthetic_source.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 扩展性实证 —— 包外加源, 不改 frame_source 一个字
"""
扩展性实证: 在 frame_source 包**之外**定义一个内存合成源,
一行 @register_source 登记自己, frame_source/ 一个字不用改。

模拟"未来某天要接的新硬件": 纯内存, 没有路径, 无法用字符串表达,
只能显式传 config —— 这正是"配置类型做派发轴"白捡的能力。

运行: python apps/platform/examples/synthetic_source.py
"""
from __future__ import annotations

import numpy as np

from od_platform.frame_source import (
    FrameSource, SourceConfig, register_source,
    Frame, FrameInfo, create_frame_source, list_sources,
)


class SyntheticConfig(SourceConfig):
    """合成源配置。纯内存, 没有路径 —— 无法用字符串表达, 只能显式传 config。"""
    n_frames: int = 30
    width:    int = 640
    height:   int = 480

    @property
    def identity(self) -> str:
        return f"synthetic:{self.n_frames}帧"


@register_source(SyntheticConfig)     # ← 无 str_matcher: config-only, 和硬件源一样只能显式选型
class SyntheticSource(FrameSource):
    """每帧现生成一张渐变测试图的内存源。"""

    def __init__(self, config: SyntheticConfig):
        super().__init__(config)
        self._n, self._w, self._h = config.n_frames, config.width, config.height
        self._i = 0

    def open(self) -> bool:
        self._i = 0
        self._frame_index = 0
        return True

    def read(self):
        if self._i >= self._n:
            return None
        img = np.full((self._h, self._w, 3), (self._i * 8) % 256, dtype=np.uint8)
        info = FrameInfo(
            width=self._w, height=self._h,
            source_type="synthetic",              # ← 第三方类型字符串(不是内置枚举成员)
            source_path=self.source_path,
            frame_index=self._i, total_frames=self._n,
            filename=f"synthetic_{self._i:04d}",
        )
        self._i += 1
        self._frame_index = self._i
        return Frame(image=img, info=info)

    def close(self) -> None:
        pass

    def get_source_type(self):
        return "synthetic"                        # ← 返回字符串(因为不是内置枚举, §2.1 的放宽在此兑现)


if __name__ == "__main__":
    # 本模块被 import 即自注册(装饰器副作用), 无需手动登记
    print("注册表:", list_sources())              # ← 多了一条 SyntheticConfig, frame_source/ 一个字没改
    with create_frame_source(SyntheticConfig(n_frames=3, width=320, height=240)) as src:
        print("modalities:", src.modalities)
        for frame in src:
            print(frame.info.frame_index, frame.image.shape, frame.info.source_type)
