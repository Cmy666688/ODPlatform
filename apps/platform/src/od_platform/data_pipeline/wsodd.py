"""WSODD VOC 原件只读接入:校验配对、按采集日分组、输出 YOLO 框与身份契约。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import yaml

from od_platform.common import paths
from od_platform.common.json_io import write_json
from od_platform.common.semantic import file_sha256
from od_platform.data_pipeline.import_dataset import extract_archive

NAMES = ["boat", "ship", "person", "bridge", "rock", "ball", "platform", "rubbish", "mast", "harbor", "animal", "buoy", "tree", "grass"]
SPLITS = ("train", "val", "test")


def fingerprint(samples, names):
    rows = sorted((row["stem"], row["split"], row["group"], row["image_sha256"], row["label_sha256"]) for row in samples)
    return hashlib.sha256(json.dumps({"names": names, "samples": rows}, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def source_group(stem, folder):
    match = re.match(r"(.*_)(\d{8})\d+$", stem)
    return match[1] + match[2] if match else "folder:" + folder


def assign_groups(rows, seed=42, names=None):
    names = NAMES if names is None else names
    groups = sorted({row["group"] for row in rows}); index = {key: i for i, key in enumerate(groups)}
    counts = np.zeros((len(groups), len(names) + 1))
    for row in rows:
        counts[index[row["group"]], 0] += 1
        for label in row["classes"]: counts[index[row["group"]], label + 1] += 1
    if len(groups) < 3: raise ValueError("独立采集分组少于三个,不能生成 train/val/test")
    totals = counts.sum(axis=0); target = np.array([0.7, 0.15, 0.15])[:, None] * totals
    rng = random.Random(seed); best = None; best_score = float("inf")
    for _ in range(12000):
        chosen = np.array(rng.choices(range(3), weights=[0.7, 0.15, 0.15], k=len(groups)))
        actual = np.array([counts[chosen == split].sum(axis=0) for split in range(3)])
        if np.any(actual[:, 0] == 0): continue
        score = float(np.sum(((actual - target) / np.maximum(target, 1)) ** 2))
        score += int(np.sum((actual[0, 1:] == 0) & (totals[1:] > 0))) * 1e9
        score += int(np.sum((actual[1:, 1:] == 0) & (totals[None, 1:] > 0))) * 1000
        score += float(np.sum(((actual[:, 0] - target[:, 0]) / target[:, 0]) ** 2)) * 50
        if score < best_score: best, best_score = chosen.copy(), score
    if best is None: raise ValueError("无法生成独立分组划分")
    return {group: SPLITS[int(best[index[group]])] for group in groups}


def prepare(archive: Path):
    incoming = paths.META_DIR / "dataset_uploads/wsodd/extracted"
    if not incoming.exists(): extract_archive(archive, incoming)
    base = incoming / "WSODD USV_dataset"
    images = {p.stem: p for p in (base / "image").glob("*.jpg")}
    annotations = {p.stem: p for p in (base / "annotation").glob("*.xml")}
    if not images or images.keys() != annotations.keys(): raise ValueError("WSODD 图像与 XML 标注不完整配对")
    root = paths.PROCESSED_DATA_DIR / "wsodd"
    if root.exists(): raise ValueError("processed/wsodd 已存在,请先核验已有契约,不要覆盖实验数据")
    rows = []; duplicates = defaultdict(list)
    for stem, image in sorted(images.items()):
        annotation = ET.fromstring(annotations[stem].read_bytes())
        with Image.open(image) as decoded:
            size = decoded.size; decoded.load()
            if decoded.getexif().get(274, 1) != 1: raise ValueError(f"图像 EXIF 方向尚未归一化: {stem}")
        width = int(annotation.findtext("size/width", "0")); height = int(annotation.findtext("size/height", "0"))
        if size != (width, height): raise ValueError(f"XML 与图像尺寸不一致: {stem}")
        labels = []; classes = []
        for item in annotation.findall("object"):
            name = item.findtext("name", "").strip()
            if name not in NAMES: raise ValueError(f"未知类别: {stem}/{name}")
            xmin, ymin, xmax, ymax = [float(item.findtext("bndbox/" + key, "nan")) for key in ("xmin", "ymin", "xmax", "ymax")]
            if not 0 <= xmin < xmax <= width or not 0 <= ymin < ymax <= height: raise ValueError(f"非法框: {stem}")
            label = NAMES.index(name); classes.append(label)
            labels.append(f"{label} {(xmin+xmax)/2/width:.9f} {(ymin+ymax)/2/height:.9f} {(xmax-xmin)/width:.9f} {(ymax-ymin)/height:.9f}")
        digest = file_sha256(image)
        row = {"stem": stem, "group": source_group(stem, annotation.findtext("folder", "unknown")),
               "classes": classes, "width": width, "height": height, "labels": "\n".join(labels) + ("\n" if labels else ""),
               "image_sha256": digest, "xml_sha256": file_sha256(annotations[stem])}
        rows.append(row); duplicates[digest].append(row)
    # 相同图像会合并采集组,避免同一张图跨集合;不删除原件。
    parent = {row["group"]: row["group"] for row in rows}
    def leader(key):
        while parent[key] != key: key = parent[key]
        return key
    for same in duplicates.values():
        for row in same[1:]: parent[leader(row["group"])] = leader(same[0]["group"])
    for row in rows: row["group"] = leader(row["group"])
    assignment = assign_groups(rows)
    for row in rows:
        split = assignment[row["group"]]; row["split"] = split
        image = root / "images" / split / (row["stem"] + ".jpg")
        label = root / "labels" / split / (row["stem"] + ".txt")
        image.parent.mkdir(parents=True, exist_ok=True); label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(images[row["stem"]], image); label.write_text(row.pop("labels"), encoding="utf-8")
        row["label_sha256"] = file_sha256(label)
    digest = fingerprint(rows, NAMES)
    manifest = {"format": "yolo_detect", "names": NAMES, "data_fingerprint": digest, "seed": 42, "images_fully_decoded": True,
                "split_policy": "camera_and_capture_day_else_source_folder; identical JPEG bytes join groups; class-balanced group search",
                "coordinate_policy": "original absolute pixel xyxy normalized to YOLO center/width/height; no clipping",
                "source_archive_sha256": file_sha256(archive), "samples": rows}
    manifest_path = root / "manifest.json"; write_json(manifest_path, manifest)
    document = {"format": "yolo_detect", "task": "detect", "path": "data/processed/wsodd",
                "train": "images/train", "val": "images/val", "test": "images/test", "names": dict(enumerate(NAMES)),
                "data_fingerprint": digest, "manifest_path": manifest_path.relative_to(paths.ROOT_DIR).as_posix()}
    paths.dataset_yaml_path("wsodd").write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True), encoding="utf-8")
    report = validate(paths.dataset_yaml_path("wsodd"))
    print(json.dumps({"splits": {s: sum(r["split"] == s for r in rows) for s in SPLITS}, "classes": NAMES,
                      "groups": len(assignment), "quality": report["overall_severity"], "fingerprint": digest}), flush=True)
    return report


def validate(yaml_path: Path, *, use_cache=False):
    from od_platform.data_validation.history import progress, save_report
    document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    root = paths.ROOT_DIR / document["path"]; manifest = json.loads((paths.ROOT_DIR / document["manifest_path"]).read_text(encoding="utf-8"))
    from od_platform.data_pipeline.detection_import import class_names
    names = class_names(document["names"])
    identifier = yaml_path.stem
    cache_path = paths.META_DIR / "data_validation" / (identifier + "_image_cache.json")
    cached = json.loads(cache_path.read_text(encoding="utf-8")) if use_cache and cache_path.exists() else {}
    entries = cached.get("images", {}) if cached.get("root") == str(root.resolve()) else {}
    refreshed = {}
    rows = []; issues = []; groups = defaultdict(set); hashes = defaultdict(set); counts: dict[str, Counter] = {s: Counter() for s in SPLITS}
    expected = {(row["split"], row["stem"]) for row in manifest["samples"]}
    total = len(manifest["samples"]); progress(identifier, 0, total)
    actual = {(s, p.stem) for s in SPLITS for p in (root / "images" / s).glob("*.jpg")}
    labels = {(s, p.stem) for s in SPLITS for p in (root / "labels" / s).glob("*.txt")}
    for index, original in enumerate(manifest["samples"]):
        row = dict(original); split, stem = row["split"], row["stem"]
        try:
            image = root / "images" / split / (stem + ".jpg"); label = root / "labels" / split / (stem + ".txt")
            status = image.stat(); key = image.relative_to(root).as_posix(); previous = entries.get(key, {})
            if use_cache and previous.get("size") == status.st_size and previous.get("mtime_ns") == status.st_mtime_ns:
                row["image_sha256"] = previous["sha256"]
            else:
                row["image_sha256"] = file_sha256(image)
                with Image.open(image) as decoded:
                    if decoded.size != (row["width"], row["height"]): issues.append(stem + ": dimensions")
                    decoded.verify()
            refreshed[key] = {"size": status.st_size, "mtime_ns": status.st_mtime_ns, "sha256": row["image_sha256"]}
            row["label_sha256"] = file_sha256(label)
            for line in label.read_text(encoding="utf-8").splitlines():
                values = [float(value) for value in line.split()]
                if len(values) != 5 or not all(np.isfinite(values)) or values[0] != int(values[0]) or not 0 <= int(values[0]) < len(names):
                    raise ValueError("非法类别/检测框")
                c, x, y, w, h = values
                if not (w > 0 and h > 0 and x-w/2 >= -1e-8 and y-h/2 >= -1e-8 and x+w/2 <= 1+1e-8 and y+h/2 <= 1+1e-8):
                    raise ValueError("越界或退化检测框")
                counts[split][int(c)] += 1
            rows.append(row); groups[row["group"]].add(split); hashes[row["image_sha256"]].add(split)
        except (OSError, ValueError) as error: issues.append(stem + ": " + str(error))
        if (index + 1) % 10 == 0 or index + 1 == total: progress(identifier, index + 1, total)
    checks = []
    def add(name, bad, summary, warning=False, **details):
        checks.append({"name": name, "severity": "WARNING" if bad and warning else "ERROR" if bad else "PASS", "summary": summary, "details": details})
    add("detection_pairing", actual != expected or labels != expected or any(not any(row[0] == s for row in expected) for s in SPLITS), f"图像/标签/契约配对: {len(actual)} 张图像, {len(labels)} 个标签;训练/验证/测试集均须非空")
    add("detection_boxes", bool(issues), f"图像解码、尺寸、类别与检测框异常: {len(issues)}", examples=issues[:20])
    add("detection_decode", not manifest.get("images_fully_decoded"), "所有原图已完整解码,当前文件通过 SHA256 与解码记录对账")
    add("detection_groups", any(len(value) > 1 for value in groups.values()), f"采集分组: {len(groups)} 组;禁止同组跨集合")
    add("detection_duplicates", any(len(value) > 1 for value in hashes.values()), "相同 JPEG 文件禁止跨集合")
    digest = fingerprint(rows, manifest["names"])
    add("detection_identity", digest != manifest["data_fingerprint"] or digest != document["data_fingerprint"] or manifest["names"] != names,
        "图像、标签、划分与类别映射 SHA256 身份校验", actual_fingerprint=digest)
    absent = {s: [names[c] for c in range(len(names)) if not counts[s][c]] for s in SPLITS}
    add("detection_train_coverage", bool(absent["train"]), "训练集类别覆盖", missing=absent["train"])
    add("detection_eval_coverage", bool(absent["val"] or absent["test"]), "验证/测试类别覆盖;缺失类别不具备独立验收证据", warning=True, missing=absent)
    inferred = manifest.get("split_policy") != "provided"
    add("detection_split_policy", inferred, "按文件名推断分组,需结合采集记录确认场景独立" if inferred else "保留用户划分;仍需结合原始采集记录确认场景独立", warning=True)
    report = {"run_id": identifier, "overall_severity": "ERROR" if any(c["severity"] == "ERROR" for c in checks) else "WARNING" if any(c["severity"] == "WARNING" for c in checks) else "PASS",
              "data_fingerprint": digest, "results": checks, "image_digest_cache": use_cache,
              "class_counts": {s: {names[c]: counts[s][c] for c in range(len(names))} for s in SPLITS}}
    if report["overall_severity"] != "ERROR": write_json(cache_path, {"root": str(root.resolve()), "images": refreshed})
    report = save_report(identifier, report); progress(identifier, total, total, phase="completed")
    return report


if __name__ == "__main__":
    prepare(paths.RAW_DATA_DIR / "wsodd/WSODD USV_dataset.zip")
