#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : stamp.py
# @Project   : ODPlatform
# @Function  : 数据流转卡 —— 每个环节在公共地留下自己那一联"章"
"""数据流转卡(traveler / 构建溯源)。

一批数据流经 D3 划分 → D4 验证 → D6 训练 → D7 评估, 每一站在一个【中立公共地】
留下自己那一联"章", 下游读章而不重做上游的活。

    runs/lineage/<data_fingerprint>/
    ├── transform.stamp.json     D3 盖
    ├── validate.stamp.json      D4 盖
    └── train.stamp.json         D6 盖

★ 目录名 = data_fingerprint(阶段 3) = 数据身份。带来两个白捡的性质:
    · 数据一变 → 指纹变 → 指向空目录 → 上游的章自动查不到 = 自动失效(无需作废机制)
    · 数据没变 → 同一指纹 → 章还在 = 下游不必重复劳动

★ 铁律(阶段 4.4): 每一方【只写自己那一联, 只主张自己亲自取证的事实】。
    盖章时 data_fingerprint 必须是本环节【自己算的】, 绝不读别人的章抄过来 ——
    转述不是验证。本模块只提供 write_stamp / read_stamp, 【故意不提供】"读回来改一改
    再存"的接口, 从 API 层面就堵死"在别人的章上加料"这条路。

★ 不加密码学签名: 威胁模型是防遗漏(忘了验/数据被手滑改), 不是防伪。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from od_platform.common import paths

logger = logging.getLogger(__name__)

# 各环节的章文件名(唯一真相, 谁盖谁读都用这里的常量, 不裸写字符串)
STAMP_TRANSFORM = "transform.stamp.json"
STAMP_VALIDATE = "validate.stamp.json"
STAMP_TRAIN = "train.stamp.json"

# 流转卡根目录
LINEAGE_DIR = paths.RUNS_DIR / "lineage"


def lineage_dir_of(data_fingerprint: str) -> Path:
    """这批数据的流转卡目录 = runs/lineage/<data_fingerprint>/。"""
    return LINEAGE_DIR / data_fingerprint


def write_stamp(
    *,
    stage_file: str,
    data_fingerprint: str,
    run_id: str,
    tool_version: str,
    payload: Optional[Dict[str, Any]] = None,
) -> Path:
    """盖一枚章 —— 把本环节那一联写进流转卡目录。

    Args:
        stage_file:       用哪个常量(STAMP_TRANSFORM / STAMP_VALIDATE / STAMP_TRAIN)
        data_fingerprint: ★ 本环节【自己算】的数据指纹 —— 绝不是读别人的章抄来的。
                          它同时决定章盖进哪个目录, 且会被写进章里(防搬家)。
        run_id:           本环节这次运行的身份, 供追回完整现场
        tool_version:     本环节盖章工具的版本(防旧章冒充新章)
        payload:          本环节特有的补充信息(如 D4 的 {"overall": "PASS", ...})

    Returns:
        写入的章文件路径。

    ★ 注意本函数【没有】"读回旧章合并"的逻辑: 每次都是全新写自己那一联。
      这是有意的 —— 从 API 层面堵死"在别人(或自己旧)的章上加料"。
    """
    d = lineage_dir_of(data_fingerprint)
    d.mkdir(parents=True, exist_ok=True)

    stamp: Dict[str, Any] = {
        "stage": stage_file.split(".")[0],       # "transform" / "validate" / "train"
        "data_fingerprint": data_fingerprint,    # ★① 章自己复述盖在哪批货上(防搬家)
        "run_id": run_id,
        "tool_version": tool_version,            # ★② 用哪个版本盖的(防旧章冒充新章)
    }
    if payload:
        stamp.update(payload)

    path = d / stage_file
    path.write_text(json.dumps(stamp, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已盖章: %s (data_fingerprint=%s)", path, data_fingerprint[:16])
    return path


def read_stamp(data_fingerprint: str, stage_file: str) -> Optional[Dict[str, Any]]:
    """读某一联章。不存在返回 None(下游据此判断"这一环盖了没")。

    ★ 读回来是给你【确认它在、看它说了什么】用的, 不是给你抄进自己的章用的(阶段 4.4)。
    """
    path = lineage_dir_of(data_fingerprint) / stage_file
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("章损坏, 读取失败 %s: %s", path, e)
        return None
