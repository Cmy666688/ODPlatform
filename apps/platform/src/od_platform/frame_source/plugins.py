#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : plugins.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : 第三方源发现 —— 扫 entry-points 并 import(触发其 @register_source)
"""
第三方源发现(可选, 生产级)。

约定 entry-point group = "frame_source.sources", 每个 entry-point 的值是一个
"import 即会执行 @register_source"的模块。load_plugins() 逐个 import 它们。

默认不自动调用 —— 由使用方在启动时显式 load_plugins() 决定何时发现插件。
"""
from __future__ import annotations

import logging
from importlib.metadata import entry_points

logger = logging.getLogger(__name__)

PLUGIN_GROUP = "frame_source.sources"


def load_plugins(group: str = PLUGIN_GROUP, *, raise_on_error: bool = False) -> list[str]:
    """
    发现并加载已安装的第三方源插件(import 它们以触发 @register_source)。

    Args:
        group          : entry-point group 名。
        raise_on_error : 某个插件加载失败时, True=抛出, False=记录并跳过(默认)。

    Returns:
        成功加载的插件名列表。
    """
    loaded: list[str] = []
    for ep in entry_points(group=group):     # Python 3.10+ 签名
        try:
            ep.load()                        # import 插件模块 → 触发其 @register_source
            loaded.append(ep.name)
            logger.info(f"已加载源插件: {ep.name}")
        except Exception as e:
            logger.error(f"加载源插件 {ep.name} 失败: {e}")
            if raise_on_error:
                raise
    return loaded
