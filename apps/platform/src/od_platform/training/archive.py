#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : archive.py
# @Project   : ODPlatform
# @Function  : 把训练产出的 best.pt 归档到 models/trained/(精选成品)
"""训练产物归档。runs/training/<run_id>/ 是过程现场, models/trained/ 是精选成品。
★ 命名规则【不在这里】—— stem 由调用方从 naming.run_stem() 拿; 本模块拿不到时钟、不知道命名规则, 永远拼不漂名字。"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from od_platform.common.paths import TRAINED_MODELS_DIR

logger = logging.getLogger(__name__)


def archive_best_weight(save_dir: Path, stem: str) -> Optional[Path]:
    """把现场里的 best.pt 归档到 models/trained/<stem>-best.pt。

    ⚠️ 复制而非移动: 现场要保持完整以便复盘。磁盘比排查时间便宜。
    ⚠️ 没有 best.pt 不抛: save=False 跑通测试本就没权重, 那是常态不是错误。
    """
    src = save_dir / "weights" / "best.pt"
    if not src.exists():
        logger.warning("现场里没有 best.pt, 跳过归档: %s", src)
        return None
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = TRAINED_MODELS_DIR / f"{stem}-best.pt"
    shutil.copy2(src, dst)
    logger.info("权重已归档: %s", dst)
    return dst
