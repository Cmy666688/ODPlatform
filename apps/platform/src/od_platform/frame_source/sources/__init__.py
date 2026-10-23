#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : __init__.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : sources 子包入口 —— import 四个源, 触发它们自注册
"""
sources 子包。

import 本子包的副作用: 执行各源模块顶层的 @register_source, 把四个内置源登记进
注册表。因此"导入 frame_source"→"导入 sources"→"四个源自注册"这条链一走完,
create_frame_source 就能派发到它们, 而 factory.py 完全不认识这四个类。
"""
from __future__ import annotations

from .camera import CameraSource
from .video  import VideoSource
from .image  import ImageSource, ImageFolderSource

__all__ = ["CameraSource", "VideoSource", "ImageSource", "ImageFolderSource"]
