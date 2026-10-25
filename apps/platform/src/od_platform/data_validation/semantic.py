#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : semantic.py
# @Project   : ODPlatform
# @Function  : 语义掩膜只读质检,复用严重度和报告出口
"""检查图像和掩膜,不把像素标签误当成 YOLO 多边形。

source_group_overlap 只基于文件名提出风险提示,不声称已证明同源泄漏。
指纹从本次实际读取的图像/掩膜重算,不能只核验标签或抄 manifest。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import time

import numpy as np

from od_platform.common import paths
from od_platform.common.run_context import RunContext
from od_platform.common.semantic import file_sha256, load_semantic_dataset, read_bgr, read_binary_mask, read_pixels, semantic_fingerprint
from od_platform.data_validation.registry import CheckResult, CheckSeverity
from od_platform.data_validation.report import ValidationReport


@dataclass
class MaskSnapshot:
    yaml_path: Path
    data_fingerprint: str


def validate_semantic_dataset(yaml_path: Path, *, run_id=None, write_report=True) -> ValidationReport:
    start = time.perf_counter()
    document, root = load_semantic_dataset(yaml_path)
    results = []
    actual = []
    duplicates = defaultdict(list)
    groups = defaultdict(set)
    missing = []; malformed = []; sizes = []; orphans: list[str] = []; alpha = []
    counts = {}; coverage = {}; nonempty = []
    for split in ("train", "val", "test"):
        image_dir, mask_dir = root / "images" / split, root / "masks" / split
        images = sorted(image_dir.glob("*.png"))
        counts[split] = len(images)
        if not images:
            missing.append(f"{split}: 没有图像")
        orphan_stems = {p.stem for p in mask_dir.glob("*.png")} - {p.stem for p in images}
        orphans.extend(f"{split}/{stem}" for stem in sorted(orphan_stems))
        fractions = []
        for image_path in images:
            mask_path = mask_dir / image_path.name
            if not mask_path.exists():
                missing.append(str(mask_path)); continue
            try:
                raw_image, raw_mask = read_pixels(image_path), read_pixels(mask_path)
                if raw_image.ndim == 3 and raw_image.shape[2] == 4:
                    alpha.append(str(image_path))
                if raw_mask.ndim == 3 and raw_mask.shape[2] == 4:
                    alpha.append(str(mask_path))
                image, mask = read_bgr(image_path), read_binary_mask(mask_path)
                if image.shape[:2] != mask.shape:
                    sizes.append(str(image_path))
                fraction = float(np.mean(mask > 0)); fractions.append(fraction)
                if fraction == 0 or fraction == 1:
                    nonempty.append({"file": str(mask_path), "foreground_fraction": fraction})
                digest = hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
                duplicates[digest].append(f"{split}/{image_path.stem}")
                group = re.sub(r" \(\d+\)$", "", image_path.stem)
                groups[group].add(split)
                actual.append({"stem": image_path.stem, "split": split,
                               "image_sha256": file_sha256(image_path), "mask_sha256": file_sha256(mask_path)})
            except (OSError, ValueError) as e:
                malformed.append({"file": str(image_path), "error": str(e)})
        coverage[split] = {"images": len(images), "mean_foreground_fraction": float(np.mean(fractions)) if fractions else None}

    def add(name, severity, summary, **details):
        results.append(CheckResult(name, severity, summary, details))

    add("mask_pair_existence", CheckSeverity.ERROR if missing or orphans else CheckSeverity.PASS,
        f"图像/掩膜配对: missing={len(missing)}, orphan={len(orphans)}", missing=missing[:20], orphans=orphans[:20])
    add("mask_decode_and_format", CheckSeverity.ERROR if malformed else CheckSeverity.PASS,
        f"无法解码/非二值掩膜: {len(malformed)}", preview=malformed[:20])
    add("mask_dimensions", CheckSeverity.ERROR if sizes else CheckSeverity.PASS,
        f"图像/掩膜尺寸不一致: {len(sizes)}", preview=sizes[:20])
    add("mask_alpha_channels", CheckSeverity.WARNING if alpha else CheckSeverity.PASS,
        f"含 alpha 通道的输入: {len(alpha)}", preview=alpha[:20], policy="忽略 alpha;仅 RGB 定义标注")
    add("mask_foreground_coverage", CheckSeverity.WARNING if nonempty else CheckSeverity.INFO,
        f"全空/全满掩膜: {len(nonempty)}", splits=coverage, preview=nonempty[:20])
    cross = [v for v in duplicates.values() if len({s.split('/')[0] for s in v}) > 1]
    add("mask_split_uniqueness", CheckSeverity.ERROR if cross else CheckSeverity.PASS,
        f"跨集合完全相同图像组: {len(cross)}", preview=cross[:20])
    overlap = {g: sorted(s) for g, s in groups.items() if len(s) > 1}
    add("mask_source_group_overlap", CheckSeverity.WARNING if overlap else CheckSeverity.PASS,
        f"同前缀文件跨集合: {len(overlap)} 组(可能同源裁剪,需结合原始采集记录确认)",
        groups=overlap, inferred=True)

    fingerprint = semantic_fingerprint(actual)
    try:
        manifest_path = paths.ROOT_DIR / document["manifest_path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = semantic_fingerprint(manifest["samples"])
        matched = fingerprint == expected == document["data_fingerprint"] == manifest["data_fingerprint"]
        add("mask_manifest_lineage", CheckSeverity.PASS if matched else CheckSeverity.ERROR,
            "图像+掩膜身份一致" if matched else "图像/掩膜/划分被改动,与契约不一致",
            actual_fingerprint=fingerprint, expected_fingerprint=expected, counts=counts)
    except (OSError, ValueError, KeyError, TypeError) as e:
        add("mask_manifest_lineage", CheckSeverity.ERROR, f"无法核验掩膜契约: {e}")

    with RunContext("data_validation") as run:
        report = ValidationReport(run_id or run.run_id, MaskSnapshot(yaml_path, fingerprint), results,
                                  time.perf_counter() - start)
        report.render_to_logger()
        if write_report:
            out = paths.validation_run_dir(report.run_id)
            report.write_json(out / "report.json"); report.write_results_csv(out / "results.csv")
    return report
