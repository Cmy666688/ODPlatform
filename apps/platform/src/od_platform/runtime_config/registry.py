#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : registry.py
# @Author    : 雨霓同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : runtime_config 子系统——"名字 → 配置类"的唯一真相
"""配置注册表: 一张共享表, 把 CLI 里的名字映射到配置类.

谁读它:
    - generator.main()  —— odp-gen-config <name> 靠它拿配置类
                           (argparse choices 也直接取自它的 keys)
    - __init__.build_*  —— 便捷构建函数保持同一份映射

为什么是一张显式 dict, 而不是 registry_utils 的自动扫描:
    自动扫描是给"实现集会无限增长、第三方会加"的场景的(D3 的 converters/).
    这里只有 3 个固定核心模式, 一张表就够 —— 加新模式 = 加一行.
    哪天真要支持第三方注册, 再换成 import_submodules, 接口不变.
"""
from __future__ import annotations

from od_platform.runtime_config.infer import YOLOInferConfig
from od_platform.runtime_config.train import YOLOTrainConfig
from od_platform.runtime_config.val   import YOLOValConfig

# 名字 → (配置类, 模板标题). ★ 加新模式 = 加一行, 分发逻辑一行不改.
CONFIG_REGISTRY: dict[str, tuple[type, str]] = {
    "train": (YOLOTrainConfig, "YOLO 训练配置"),
    "val":   (YOLOValConfig,   "YOLO 验证配置"),
    "infer": (YOLOInferConfig, "YOLO 推理配置"),
}

__all__ = ["CONFIG_REGISTRY"]
