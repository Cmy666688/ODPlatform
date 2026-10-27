"""训练引擎可以修复副本,但不能改写冻结图像或复用标签缓存。"""
import json

import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError
import yaml

from od_platform.runtime_config.detection import DetectionConfig
from od_platform.training.detection_data import stage_dataset


def test_staging_repairs_jpeg_eoi_without_changing_source_pixels_or_labels(tmp_path):
    root = tmp_path / "original"; job = tmp_path / "job"; job.mkdir()
    image = root / "images/train/a.jpg"; image.parent.mkdir(parents=True)
    Image.new("RGB", (40, 30), (24, 80, 120)).save(image)
    # 可完整解码但最后两字节不是 EOI 的合法源件,模拟引擎会重新保存的情况。
    original = image.read_bytes() + b"trailing JPEG bytes"; image.write_bytes(original)
    labels = root / "labels/train/a.txt"; labels.parent.mkdir(parents=True); labels.write_text("0 .5 .5 .2 .2\n")
    (root / "labels/train.cache").write_bytes(b"stale cache")
    document = {"train": "images/train", "val": "images/val", "test": "images/test", "names": {0:"ship"}, "data_fingerprint":"frozen"}
    events = []; data, audit = stage_dataset(document, root, job, events.append)
    copy = job / "engine_dataset/images/train/a.jpg"
    assert image.read_bytes() == original and copy.read_bytes() == original + b"\xff\xd9"
    assert np.array_equal(np.asarray(Image.open(image)), np.asarray(Image.open(copy)))
    assert not (job / "engine_dataset/labels/train.cache").exists()
    assert (job / "engine_dataset/labels/train/a.txt").read_bytes() == labels.read_bytes()
    assert yaml.safe_load(data.read_text())["names"] == document["names"]
    assert json.loads((job / "engine_dataset_audit.json").read_text())["data_fingerprint"] == "frozen"
    assert len(audit["jpeg_eoi_appended"]) == 1 and events[-1]["staged_images"] == 1
    copy.write_bytes(b"engine changed pixels")
    assert image.read_bytes() == original
    with pytest.raises(ValueError): stage_dataset(document, root, job, events.append)


def test_improvement_config_is_bounded_and_keeps_old_baseline_defaults():
    baseline = DetectionConfig(); assert baseline.architecture == "yolov8n.pt" and baseline.mosaic == 1
    improved = DetectionConfig(architecture="yolov8s.pt", imgsz=960, cos_lr=True, warmup_bias_lr=.01, scale=.3, mosaic=.5)
    assert improved.device == "cuda" and improved.imgsz == 960
    with pytest.raises(ValidationError): DetectionConfig(architecture="../../custom.pt")
    with pytest.raises(ValidationError): DetectionConfig(mosaic=2)


def test_fast_recipe_keeps_full_data_and_exposes_performance_options():
    from od_platform.common import paths
    from od_platform.runtime_config.train import YOLOTrainConfig
    fast = DetectionConfig(**yaml.safe_load(paths.runtime_config_path("wsodd_fast_4090").read_text(encoding="utf-8")))
    assert fast.architecture == "yolov8s.pt" and fast.imgsz == 640 and fast.epochs == 30
    assert fast.batch_size_train == 64 and fast.num_threads == 8 and fast.cache == "ram"
    assert fast.rect and not fast.deterministic and fast.max_train_batches is None
    engine = YOLOTrainConfig(cache=fast.cache, rect=fast.rect, deterministic=fast.deterministic).to_ultralytics_kwargs()
    assert engine["cache"] == "ram" and engine["rect"] is True and engine["deterministic"] is False
    with pytest.raises(ValidationError): DetectionConfig(cache="arbitrary")
    with pytest.raises(ValidationError): DetectionConfig(num_threads=128)
