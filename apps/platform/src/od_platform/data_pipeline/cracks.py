#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : cracks.py
# @Project   : ODPlatform
# @Function  : MixerCSeg 掩膜数据接入,保留上游 train/val/test
"""仅写新的 processed/<dataset> 叶子,raw 只读。原始 archive 保留用于复核。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re

import yaml

from od_platform.common import paths
from od_platform.common.constants import IMAGE_EXTENSIONS
from od_platform.common.run_context import RunContext
from od_platform.common.semantic import file_sha256, read_bgr, read_binary_mask, semantic_fingerprint, write_png
from od_platform.runtime_config.mixercseg import UPSTREAM_COMMIT

ARCHIVES = {
    "CrackMap": ("1efIsca3YKwQsQS1AOBg98KZwYrg_Ug-p", "2abcae4379e6fb8c5907b6eefc14908037eb225073b70ef5760ef9a15ba316c0"),
    "DeepCrack": ("17FpAsFoS10rIoEpaw2RuuRRDQEvEVplX", "3150d3c65b8145df1f93fe706086b1f922300d8923ef849517fdbafb6b74250f"),
}


def prepare_cracks(source: Path, dataset: str = "crackmap") -> Path:
    """RGBA → RGB/L,不改变尺寸、二值前景或官方划分。存在目标时拒绝覆盖。"""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", dataset):
        raise ValueError(f"非法数据集名: {dataset!r}")
    target = paths.dataset_processed_dir(dataset)
    if target.exists():
        raise FileExistsError(f"派生数据集已存在,请使用新名称避免覆盖: {target}")
    samples = []
    counts: Counter[str] = Counter()
    # 先把所有输入验完,再开始物化,不能留下半套可训练数据。
    inputs = []
    for split in ("train", "val", "test"):
        image_dir, mask_dir = source / f"{split}_img", source / f"{split}_lab"
        images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"划分没有图像: {image_dir}")
        if len({p.stem for p in images}) != len(images):
            raise ValueError(f"同名图像冲突: {image_dir}")
        orphans = {p.stem for p in mask_dir.glob("*.png")} - {p.stem for p in images}
        if orphans:
            raise ValueError(f"孤儿掩膜: {mask_dir}: {sorted(orphans)[:5]}")
        for image_path in images:
            mask_path = mask_dir / (image_path.stem + ".png")
            image, mask = read_bgr(image_path), read_binary_mask(mask_path)
            if image.shape[:2] != mask.shape:
                raise ValueError(f"图像/掩膜尺寸不匹配: {image_path}")
            inputs.append((split, image_path, image, mask))

    with RunContext("data_pipeline") as run:
        for split, original, image, mask in inputs:
            image_path = target / "images" / split / (original.stem + ".png")
            mask_path = target / "masks" / split / (original.stem + ".png")
            write_png(image_path, image); write_png(mask_path, mask)
            samples.append({"stem": original.stem, "split": split,
                            "image_sha256": file_sha256(image_path), "mask_sha256": file_sha256(mask_path)})
            counts[split] += 1
        fingerprint = semantic_fingerprint(samples)
        manifest = {"schema_version": 1, "format": "semantic_masks", "samples": samples,
                    "data_fingerprint": fingerprint, "upstream_commit": UPSTREAM_COMMIT,
                    "created_at": run.created_at, "counts": dict(counts)}
        manifest_path = run.artifact_path("semantic_manifest.json")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        document = {"format": "semantic_masks", "task": "semantic_segment",
                    "path": target.relative_to(paths.ROOT_DIR).as_posix(),
                    "names": {0: "background", 1: "crack"},
                    "splits": ["train", "val", "test"],
                    "source": {"repository": "https://github.com/spiderforest/MixerCSeg",
                               "commit": UPSTREAM_COMMIT, "dataset": source.name,
                               "alpha_policy": "discard; RGB image and single-channel binary mask"},
                    "manifest_path": manifest_path.relative_to(paths.ROOT_DIR).as_posix(),
                    "data_fingerprint": fingerprint}
        yaml_path = paths.dataset_yaml_path(dataset)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return yaml_path
