"""用户数据集接入:受限解包、布局识别、固定划分与二值掩膜契约。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import stat
import tarfile
import zipfile
from uuid import uuid4

import yaml

from od_platform.common import paths
from od_platform.common.json_io import write_json
from od_platform.common.run_context import RunContext
from od_platform.common.semantic import file_sha256, read_bgr, read_binary_mask, semantic_fingerprint, write_png

MAX_FILES = 20000
MAX_EXPANDED = 2 * 1024**3
MAX_FILE = 64 * 1024**2
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def safe_relative(name: str) -> Path:
    normalized = name.replace("\\", "/")
    while normalized.startswith("./"): normalized = normalized[2:]
    value = PurePosixPath(normalized)
    if (not normalized or value.is_absolute() or any(ord(char) < 32 for char in normalized) or any(part in (".", "..") for part in normalized.split("/"))
            or any(":" in part or part.endswith((".", " ")) for part in value.parts)
            or any(re.match(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", part, re.I) for part in value.parts)):
        raise ValueError(f"不安全的文件路径: {name}")
    return Path(*value.parts)


def extract_archive(archive: Path, destination: Path, *, max_expanded=None, max_files=None, on_progress=None) -> None:
    """不跟随链接,同时核对声明大小和实际读取大小。"""
    expanded_limit = MAX_EXPANDED if max_expanded is None else max_expanded
    file_limit = MAX_FILES if max_files is None else max_files
    destination.mkdir(parents=True, exist_ok=False)
    seen = set(); expanded = 0; count = 0; total = 0
    def materialize(name, size, stream):
        nonlocal expanded, count
        relative = safe_relative(name)
        key = relative.as_posix().casefold()
        count += 1
        if key in seen or count > file_limit or size > MAX_FILE or expanded + size > expanded_limit:
            raise ValueError("压缩包存在重复路径或超过文件数/解压大小限制")
        seen.add(key)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        actual = 0
        with target.open("xb") as output:
            while chunk := stream.read(1024 * 1024):
                actual += len(chunk); expanded += len(chunk)
                if actual > MAX_FILE or expanded > expanded_limit or actual > size:
                    raise ValueError("解压数据超过声明大小或限制")
                output.write(chunk)
        if actual != size:
            raise ValueError("压缩包成员大小不完整")
        if on_progress: on_progress(count, total)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            total = sum(not member.is_dir() for member in source.infolist())
            if len(source.infolist()) > file_limit * 2: raise ValueError("压缩包文件项过多")
            for member in source.infolist():
                safe_relative(member.filename.rstrip("/"))
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or member.flag_bits & 1:
                    raise ValueError("不支持链接或加密压缩包")
                if member.is_dir(): continue
                with source.open(member) as stream:
                    materialize(member.filename, member.file_size, stream)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tar_source:
            for index, tar_member in enumerate(tar_source):
                if index > file_limit * 2: raise ValueError("压缩包文件项过多")
                if tar_member.isdir() and tar_member.name in (".", "./"): continue
                safe_relative(tar_member.name.rstrip("/"))
                if tar_member.isdir(): continue
                if not tar_member.isfile(): raise ValueError("压缩包仅允许普通文件和目录")
                tar_stream = tar_source.extractfile(tar_member)
                if tar_stream is None: raise ValueError("无法读取压缩包成员")
                with tar_stream:
                    materialize(tar_member.name, tar_member.size, tar_stream)
    else:
        raise ValueError("请上传 ZIP、TAR、TAR.GZ 或 TGZ 压缩包")
    if on_progress: on_progress(count, count)


def _pairs(image_dir: Path, mask_dir: Path) -> list:
    def index(directory):
        result = {}
        if any(path.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES for p in path.rglob("*") if p.is_file()) for path in directory.iterdir()):
            raise ValueError(f"图像与掩膜目录须平铺放置文件,不支持嵌套子目录: {directory}")
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES: continue
            key = path.stem.casefold()
            if key in result: raise ValueError(f"同名图片冲突: {directory}/{path.stem}")
            result[key] = path
        return result
    images, masks = index(image_dir), index(mask_dir)
    if not images or images.keys() != masks.keys():
        missing = sorted(images.keys() - masks.keys())[:5]
        orphan = sorted(masks.keys() - images.keys())[:5]
        raise ValueError(f"图像与掩膜须按文件名一一配对;缺少标注={missing},孤立标注={orphan}")
    return [(path, masks[key]) for key, path in images.items()]


def discover_samples(source: Path, seed=42) -> tuple[dict, str]:
    candidates = []
    for root in [source, *sorted(p for p in source.rglob("*") if p.is_dir() and "__MACOSX" not in p.parts)]:
        layouts = [
            {s: (root / "images" / s, root / "masks" / s) for s in ("train", "val", "test")},
            {s: (root / (s + "_img"), root / (s + "_lab")) for s in ("train", "val", "test")},
            {s: (root / s / "images", root / s / "masks") for s in ("train", "val", "test")},
        ]
        for layout in layouts:
            if all(i.is_dir() and m.is_dir() for i, m in layout.values()):
                candidates.append(({s: _pairs(i, m) for s, (i, m) in layout.items()}, "provided"))
        images, masks = root / "images", root / "masks"
        if images.is_dir() and masks.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES for p in images.iterdir() if p.is_file()):
            groups = defaultdict(list)
            for image, mask in _pairs(images, masks):
                group = re.sub(r" \(\d+\)$", "", image.stem).casefold()
                groups[group].append((image, mask))
            names = sorted(groups); random.Random(seed).shuffle(names)
            if len(names) < 3: raise ValueError("自动划分至少需要 3 个独立文件名前缀组;也可提供 train/val/test")
            test_n = max(1, round(len(names) * 0.15)); val_n = max(1, round(len(names) * 0.15))
            assignment = {"test": names[:test_n], "val": names[test_n:test_n + val_n], "train": names[test_n + val_n:]}
            candidates.append(({s: [pair for name in subset for pair in groups[name]] for s, subset in assignment.items()}, "grouped_seed42"))
    if len(candidates) != 1:
        raise ValueError("未找到唯一数据集。支持 images/masks、images/{train,val,test}+masks/{train,val,test}、train_img/train_lab 等结构")
    return candidates[0]


def import_semantic_dataset(source: Path, dataset: str, *, on_progress=None) -> Path:
    if not re.fullmatch(r"ds_[0-9a-f]{16}", dataset): raise ValueError("非法导入数据集 ID")
    samples, strategy = discover_samples(source)
    target = paths.dataset_processed_dir(dataset)
    if target.exists(): raise FileExistsError("数据集已存在,不能覆盖")
    staging = target.with_name(".import-" + uuid4().hex)
    staging.mkdir(parents=True, exist_ok=False)
    total = sum(len(rows) for rows in samples.values()); records = []
    try:
        for split, rows in samples.items():
            for original, annotation in rows:
                from PIL import Image
                for file in (original, annotation):
                    with Image.open(file) as header:
                        if header.width * header.height > 16_000_000: raise ValueError(f"图像尺寸过大: {file.name}")
                image, mask = read_bgr(original), read_binary_mask(annotation)
                if image.shape[:2] != mask.shape: raise ValueError(f"图像与掩膜尺寸不同: {original.name}")
                if image.size > 48_000_000: raise ValueError(f"图像尺寸过大: {original.name}")
                filename = original.stem + ".png"
                image_path, mask_path = staging / "images" / split / filename, staging / "masks" / split / filename
                write_png(image_path, image); write_png(mask_path, mask)
                records.append({"stem": original.stem, "split": split,
                                "image_sha256": file_sha256(image_path), "mask_sha256": file_sha256(mask_path)})
                if on_progress: on_progress(len(records), total)
        fingerprint = semantic_fingerprint(records)
        with RunContext("data_pipeline") as run:
            manifest_path = run.artifact_path("semantic_manifest.json")
            counts = {s: len(rows) for s, rows in samples.items()}
            write_json(manifest_path, {"schema_version": 1, "format": "semantic_masks", "samples": records,
                       "data_fingerprint": fingerprint, "counts": counts, "split_strategy": strategy, "seed": 42})
            document = {"format": "semantic_masks", "task": "semantic_segment", "names": {0: "background", 1: "foreground"},
                        "path": target.relative_to(paths.ROOT_DIR).as_posix(), "data_fingerprint": fingerprint,
                        "manifest_path": manifest_path.relative_to(paths.ROOT_DIR).as_posix(),
                        "source": {"kind": "user_upload", "split_strategy": strategy, "seed": 42}}
            target.parent.mkdir(parents=True, exist_ok=True); staging.rename(target)
            yaml_path = paths.dataset_yaml_path(dataset); yaml_path.parent.mkdir(parents=True, exist_ok=True)
            yaml_path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return yaml_path
    finally:
        # 只回收本次函数创建的临时叶子,不删除任何已有数据集。
        if staging.exists() and staging.parent.resolve() == target.parent.resolve() and staging.name.startswith(".import-"):
            shutil.rmtree(staging)
