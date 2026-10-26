"""目标检测数据接入:识别 VOC/YOLO,保留类别,输出可追溯的检测框契约。"""
from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
from uuid import uuid4

import numpy as np
from PIL import Image
import yaml

from od_platform.common import paths
from od_platform.common.json_io import write_json
from od_platform.common.semantic import file_sha256
from od_platform.data_pipeline.import_dataset import IMAGE_SUFFIXES
from od_platform.data_pipeline.wsodd import SPLITS, assign_groups, fingerprint

MAX_UPLOAD = 4 * 1024**3
MAX_EXPANDED = 12 * 1024**3
MAX_FILES = 100000


def class_names(value):
    if isinstance(value, dict):
        if set(value) != set(range(len(value))): raise ValueError("类别 ID 必须从 0 开始连续编号")
        value = [value[i] for i in range(len(value))]
    if (not isinstance(value, list) or not 1 <= len(value) <= 1000
            or any(not isinstance(name, str) or not name.strip() or len(name) > 100 for name in value)
            or len(set(value)) != len(value)):
        raise ValueError("请提供不重复的类别名称列表,支持 1 至 1000 个目标类别")
    return value


def read_names(root):
    candidates = [p for p in root.iterdir() if p.name.lower() in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml", "classes.txt")]
    if len(candidates) != 1: raise ValueError("YOLO 数据须包含唯一 data.yaml（names）或 classes.txt 类别表")
    file = candidates[0]
    if file.stat().st_size > 1024**2: raise ValueError("类别配置文件过大")
    if file.suffix == ".txt": return class_names(file.read_text(encoding="utf-8-sig").splitlines())
    value = yaml.safe_load(file.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict): raise ValueError("data.yaml 须包含 names 类别表")
    names = class_names(value.get("names"))
    if "nc" in value and value["nc"] != len(names): raise ValueError("nc 与 names 类别数量不一致")
    # 上传配置中的路径、下载脚本及训练参数不进入服务端运行配置。
    return names


def paired(images, labels, suffix):
    def index(directory, image):
        result = {}
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or (path.suffix.lower() not in IMAGE_SUFFIXES if image else path.suffix.lower() != suffix): continue
            if path.is_symlink(): raise ValueError("不支持符号链接")
            key = path.relative_to(directory).with_suffix("").as_posix().casefold()
            if key in result: raise ValueError("同名文件冲突: " + key)
            result[key] = path
        return result
    original, annotation = index(images, True), index(labels, False)
    if not original or original.keys() != annotation.keys():
        raise ValueError(f"图像与标注必须同名配对;缺失={sorted(original.keys()-annotation.keys())[:5]},孤立={sorted(annotation.keys()-original.keys())[:5]}")
    return [(original[key], annotation[key]) for key in original]


def discover(source):
    candidates = []
    roots = [source, *sorted(p for p in source.rglob("*") if p.is_dir() and p.name not in ("__MACOSX",))]
    for root in roots:
        for image_dir, label_dir, kind in (("JPEGImages", "Annotations", "voc"), ("image", "annotation", "voc"), ("images", "annotations", "voc"), ("images", "labels", "yolo")):
            images, labels = root / image_dir, root / label_dir
            if not images.is_dir() or not labels.is_dir(): continue
            suffix = ".xml" if kind == "voc" else ".txt"
            if any((images / s).is_dir() or (labels / s).is_dir() for s in SPLITS):
                if not all((images / s).is_dir() and (labels / s).is_dir() for s in SPLITS):
                    raise ValueError("已有划分须同时提供 train、val、test 及对应标签,独立测试集不能为空")
                samples = {s: paired(images / s, labels / s, suffix) for s in SPLITS}; strategy = "provided"
            else:
                samples = {"unsplit": paired(images, labels, suffix)}; strategy = "grouped_seed42"
                lists = root / "ImageSets/Main"
                if kind == "voc" and lists.exists():
                    if not all((lists / (s + ".txt")).is_file() for s in SPLITS): raise ValueError("VOC ImageSets/Main 需提供 train.txt、val.txt、test.txt")
                    by_stem = {i.stem: (i, a) for i, a in samples["unsplit"]}
                    if len(by_stem) != len(samples["unsplit"]): raise ValueError("VOC 划分中的文件名不唯一")
                    samples = {}; seen = set()
                    for split in SPLITS:
                        ids = (lists / (split + ".txt")).read_text(encoding="utf-8-sig").split()
                        if not ids or len(ids) != len(set(ids)) or seen.intersection(ids) or set(ids) - by_stem.keys():
                            raise ValueError("VOC 划分为空、重复、跨集合重叠或引用不存在的图片")
                        samples[split] = [by_stem[key] for key in ids]; seen.update(ids)
                    if seen != by_stem.keys(): raise ValueError("VOC 划分未覆盖所有图像,请整理后上传")
                    strategy = "provided"
            candidates.append((root, kind, samples, strategy))
        if any((root / s / "images").is_dir() for s in SPLITS):
            if not all((root / s / "images").is_dir() and (root / s / "labels").is_dir() for s in SPLITS):
                raise ValueError("已有 YOLO 划分须同时提供 train/val/test 的 images 与 labels")
            candidates.append((root, "yolo", {s: paired(root / s / "images", root / s / "labels", ".txt") for s in SPLITS}, "provided"))
    # split/images+labels 在父目录已识别,不再把各 split 当作独立数据集。
    candidates = [item for item in candidates if not any(item[0] == other[0] / s for other in candidates for s in SPLITS)]
    if len(candidates) != 1: raise ValueError("未找到唯一 VOC 或 YOLO 目标检测数据集,请参考页面格式说明;不接受掩膜或分类文件夹")
    return candidates[0]


def group_name(stem):
    dated = re.match(r"(.*_)(\d{8})\d+$", stem)
    return dated[1] + dated[2] if dated else re.sub(r"[_ -]\d+$", "", stem).casefold()


def boxes(text, count):
    result = []
    for line in text.splitlines():
        values = [float(v) for v in line.split()]
        if len(values) != 5 or not all(np.isfinite(values)): raise ValueError("YOLO 标签每行须为 class x_center y_center width height 五列有限数值")
        c, x, y, w, h = values
        if c != int(c) or not 0 <= c < count: raise ValueError("类别 ID 超出类别表范围")
        if not (w > 0 and h > 0 and x-w/2 >= -1e-8 and y-h/2 >= -1e-8 and x+w/2 <= 1+1e-8 and y+h/2 <= 1+1e-8):
            raise ValueError("检测框越界或退化,须使用归一化坐标")
        result.append(values)
    return result


def import_detection_dataset(source: Path, dataset: str, *, on_progress=None) -> Path:
    if not re.fullmatch(r"ds_[0-9a-f]{16}", dataset): raise ValueError("非法导入数据集 ID")
    base, kind, samples, strategy = discover(source)
    names = read_names(base) if kind == "yolo" else []
    rows = []; sources = {}; annotations = {}
    total = sum(len(value) for value in samples.values())
    for split, values in samples.items():
        for image, label in values:
            stem = image.stem + "_" + hashlib.sha256(image.relative_to(base).as_posix().encode()).hexdigest()[:10]
            with Image.open(image) as decoded:
                width, height = decoded.size
                if width * height > 16_000_000: raise ValueError("图像超过 1600 万像素: " + image.name)
                if decoded.getexif().get(274, 1) != 1: raise ValueError("请先归一化 EXIF 方向并同步标注: " + image.name)
                decoded.load()
            if label.stat().st_size > 1024**2: raise ValueError("单图标注文件过大")
            if kind == "voc":
                raw = label.read_bytes()
                if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper(): raise ValueError("XML 不允许实体或文档类型定义")
                tree = ET.fromstring(raw)
                if (int(tree.findtext("size/width", "0")), int(tree.findtext("size/height", "0"))) != (width, height):
                    raise ValueError("XML 与图像尺寸不同: " + image.name)
                objects = []
                for item in tree.findall("object"):
                    name = item.findtext("name", "").strip()
                    if not name: raise ValueError("VOC 目标缺少类别名称")
                    if name not in names: names.append(name)
                    xmin, ymin, xmax, ymax = [float(item.findtext("bndbox/" + key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
                    if not 0 <= xmin < xmax <= width or not 0 <= ymin < ymax <= height: raise ValueError("非法 VOC 检测框: " + image.name)
                    objects.append([name, (xmin+xmax)/2/width, (ymin+ymax)/2/height, (xmax-xmin)/width, (ymax-ymin)/height])
                annotations[stem] = objects
            else: annotations[stem] = boxes(label.read_text(encoding="utf-8-sig"), len(names))
            rows.append({"stem": stem, "source": image.relative_to(base).as_posix(), "split": split,
                         "group": group_name(image.stem), "width": width, "height": height,
                         "source_image_sha256": file_sha256(image), "source_label_sha256": file_sha256(label)})
            sources[stem] = image
            if on_progress: on_progress(len(rows), total * 2)
    names = class_names(sorted(names) if kind == "voc" else names)
    for row in rows:
        values = annotations[row["stem"]]
        if kind == "voc": values = [[names.index(value[0]), *value[1:]] for value in values]
        annotations[row["stem"]] = values
        row["classes"] = [int(value[0]) for value in values]
    if strategy != "provided":
        # 相同原图将其来源组联结,不随机把相邻帧或重复图像分到不同集合。
        parent = {row["group"]: row["group"] for row in rows}; hashes = defaultdict(list)
        def leader(key):
            while parent[key] != key: key = parent[key]
            return key
        for row in rows: hashes[row["source_image_sha256"]].append(row)
        for same in hashes.values():
            for row in same[1:]: parent[leader(row["group"])] = leader(same[0]["group"])
        for row in rows: row["group"] = leader(row["group"])
        assignment = assign_groups(rows, names=names)
        for row in rows: row["split"] = assignment[row["group"]]
    target = paths.dataset_processed_dir(dataset); staging = target.with_name(".import-" + uuid4().hex)
    if target.exists(): raise FileExistsError("数据集已存在,不能覆盖")
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for index, row in enumerate(rows):
            image = staging / "images" / row["split"] / (row["stem"] + ".jpg")
            label = staging / "labels" / row["split"] / (row["stem"] + ".txt")
            image.parent.mkdir(parents=True, exist_ok=True); label.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(sources[row["stem"]]) as decoded: decoded.convert("RGB").save(image, quality=95)
            label.write_text("".join(" ".join(str(v) for v in values) + "\n" for values in annotations[row["stem"]]), encoding="utf-8")
            row["image_sha256"], row["label_sha256"] = file_sha256(image), file_sha256(label)
            if on_progress: on_progress(total + index + 1, total * 2)
        digest = fingerprint(rows, names)
        write_json(staging / "manifest.json", {"format": "yolo_detect", "names": names, "samples": rows,
                   "data_fingerprint": digest, "images_fully_decoded": True, "source_format": kind,
                   "split_policy": strategy, "coordinate_policy": "VOC original pixel xyxy / YOLO normalized xywh; no clipping; RGB JPEG quality 95", "seed": 42})
        document = {"format": "yolo_detect", "task": "detect", "path": target.relative_to(paths.ROOT_DIR).as_posix(),
                    **{s: "images/" + s for s in SPLITS}, "names": dict(enumerate(names)), "data_fingerprint": digest,
                    "manifest_path": (target / "manifest.json").relative_to(paths.ROOT_DIR).as_posix()}
        staging.rename(target)
        yaml_path = paths.dataset_yaml_path(dataset); yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(yaml.safe_dump(document, allow_unicode=True, sort_keys=False), encoding="utf-8")
        return yaml_path
    finally:
        if staging.exists() and staging.parent.resolve() == target.parent.resolve() and staging.name.startswith(".import-"):
            shutil.rmtree(staging)
