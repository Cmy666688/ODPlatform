#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : naming.py
# @Project   : ODPlatform
# @Function  : 一次运行的产物命名 —— 规则只此一份
"""运行产物的命名规则(单一真相)。

为什么住 common/ 而不是 training/: 日志(get_logger)和归档(archive)是两处代码,
却必须拼出同一个名字。规则各写各的今天对上明天就漂。规则住公共层, 两边都调它,
于是【同名不再是纪律, 是必然】。

规则(从左到右, 信息量递减): <阶段>-<run_id>-<数据集>-<模型>
  · 阶段最前: 日志按 log_type 分目录, 目录内 <阶段>-<时间戳> 开头 = 天然按时间排序
  · run_id 永远第二位: grep 的钥匙, 三个产物靠它串起来
  · 全程一种连接符 '-': run_id 内部就是 20260718-121232, 混用 '_' 让眼睛来回切
"""
from __future__ import annotations

from pathlib import PurePath
from typing import Optional


def sanitize(text: str) -> str:
    """文件名安全化: 只留字母数字和 _ -, 其余换成 _。"""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in text)


def model_stem(model: str) -> str:
    """模型引用取干净主干: 'models/pretrained/yolov8n.pt' -> 'yolov8n'。"""
    return sanitize(PurePath(model).stem)


def run_stem(*, stage: str, run_id: str,
             dataset: Optional[str] = None, model: Optional[str] = None) -> str:
    """一次运行的完整身份 stem。日志、归档、任何扁平堆放的产物共用它。

    Returns: 'train-20260718-121232-MRI_PASCAL-yolov8n'

    ★ 本模块【没有 import datetime】—— 故意的。给 run_id 一个"留空就自己取表"的默认值
      会很"贴心", 但那等于把裂缝 A 的枪重新上膛。让这个模块【根本拿不到时钟】,
      它就永远不可能造出第二个真相。能力上的剥夺, 比文档里写"请别自己 now()"可靠得多。
    """
    parts = [sanitize(stage.replace("_", "-")), run_id]
    if dataset:
        parts.append(sanitize(dataset))
    if model:
        parts.append(model_stem(model))
    return "-".join(parts)
