#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : prepare_cracks.py
# @Project   : ODPlatform
# @Function  : 下载核验 + 安全解压 + RGBA 标准化 + 掩膜质检
"""odp-prepare-cracks。下载的原始数据保留,不会清理/覆盖已有数据集。"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

from od_platform.common import paths
from od_platform.common.logging_utils import get_logger
from od_platform.common.semantic import file_sha256
from od_platform.data_pipeline.cracks import ARCHIVES, prepare_cracks
from od_platform.data_validation.semantic import validate_semantic_dataset


def download_cracks(name: str) -> Path:
    import gdown
    file_id, digest = ARCHIVES[name]
    base = paths.RAW_DATA_DIR / "mixercseg"
    archive = base / "archives" / (name + ".zip")
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        gdown.download(id=file_id, output=str(archive), use_cookies=False)
    if file_sha256(archive) != digest:
        raise ValueError(f"原始压缩包 SHA256 不一致: {archive}")
    target = base / name
    if target.exists():
        return target
    with zipfile.ZipFile(archive) as data:
        for item in data.infolist():
            output = (target / item.filename).resolve()
            if (not output.is_relative_to(target.resolve()) or ":" in item.filename
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f"拒绝不安全压缩包成员: {item.filename}")
        data.extractall(target)
    return target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="odp-prepare-cracks")
    parser.add_argument("--dataset", choices=ARCHIVES, default="CrackMap")
    parser.add_argument("--name", default="crackmap")
    parser.add_argument("--source", type=Path, default=None, help="已有数据根;不传则下载原始数据")
    args = parser.parse_args(argv)
    log = get_logger(paths.LOGGING_DIR, log_type="prepare_cracks")
    try:
        source = args.source if args.source is not None else download_cracks(args.dataset)
        yaml_path = prepare_cracks(source, args.name)
        report = validate_semantic_dataset(yaml_path)
        log.info("数据集已准备: %s | 质检: %s", yaml_path, report.overall_severity)
        return 2 if report.overall_severity == "ERROR" else 0
    except Exception as e:
        log.exception("裂缝数据准备失败: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
