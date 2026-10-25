"""语义数据回归:真实 RGBA 问题、像素身份、划分泄漏与训练门禁。"""
from __future__ import annotations

import json

import numpy as np
import pytest

from od_platform.common import paths
from od_platform.common.run_context import RunContext
from od_platform.common.semantic import file_sha256, read_bgr, read_binary_mask, write_png
from od_platform.data_pipeline.cracks import prepare_cracks
from od_platform.data_validation.semantic import validate_semantic_dataset
from od_platform.data_validation.service import validate_dataset
from od_platform.runtime_config.mixercseg import MixerCSegConfig


@pytest.mark.parametrize("content", [None, "names: [unterminated", b"\xff\xfe"])
def test_invalid_yaml_still_produces_validation_report(tmp_path, content):
    yaml_path = tmp_path / "invalid.yaml"
    if isinstance(content, str):
        yaml_path.write_text(content, encoding="utf-8")
    elif isinstance(content, bytes):
        yaml_path.write_bytes(content)
    report = validate_dataset(yaml_path, write_report=False)
    assert report.overall_severity == "ERROR"


@pytest.fixture
def semantic_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(paths, "VALIDATION_RUNS_DIR", tmp_path / "runs/data_validation")
    monkeypatch.setattr(paths, "PROCESSED_DATA_DIR", tmp_path / "data/processed")
    monkeypatch.setattr(paths, "DATASET_CONFIGS_DIR", tmp_path / "configs/datasets")
    monkeypatch.setattr(paths, "TRAINED_MODELS_DIR", tmp_path / "models/trained")
    source = tmp_path / "raw/CrackMap"
    for index, split in enumerate(("train", "val", "test")):
        image = np.full((16, 16, 4), 30 + index * 50, dtype=np.uint8)
        image[..., 3] = 255
        mask = np.zeros((16, 16, 4), dtype=np.uint8)
        mask[:, 3:5, :3] = 255; mask[..., 3] = 255
        write_png(source / f"{split}_img/{split}.png", image)
        write_png(source / f"{split}_lab/{split}.png", mask)
    return source


def test_alpha_is_not_foreground_and_raw_is_unchanged(semantic_workspace):
    source = semantic_workspace
    raw_mask = source / "train_lab/train.png"
    before = file_sha256(raw_mask)
    yaml_path = prepare_cracks(source)
    assert file_sha256(raw_mask) == before
    target = paths.dataset_processed_dir("crackmap")
    assert read_binary_mask(target / "masks/train/train.png").astype(bool).sum() == 32
    assert read_bgr(target / "images/train/train.png").shape == (16, 16, 3)
    report = validate_semantic_dataset(yaml_path, write_report=False)
    assert report.overall_severity in ("PASS", "INFO")


def test_image_tamper_invalidates_identity_even_when_mask_is_unchanged(semantic_workspace):
    yaml_path = prepare_cracks(semantic_workspace)
    image_path = paths.dataset_processed_dir("crackmap") / "images/train/train.png"
    image = read_bgr(image_path); image[0, 0, 0] += 1; write_png(image_path, image)
    report = validate_semantic_dataset(yaml_path, write_report=False)
    assert next(r for r in report.results if r.name == "mask_manifest_lineage").severity == "ERROR"


def test_nonbinary_mask_fails_before_materialization(semantic_workspace):
    path = semantic_workspace / "train_lab/train.png"
    mask = read_binary_mask(path); mask[0, 0] = 128; write_png(path, mask)
    with pytest.raises(ValueError, match="二值"):
        prepare_cracks(semantic_workspace)
    assert not paths.dataset_processed_dir("crackmap").exists()


def test_preparation_never_overwrites_existing_dataset(semantic_workspace):
    prepare_cracks(semantic_workspace)
    with pytest.raises(FileExistsError):
        prepare_cracks(semantic_workspace)


def test_cross_split_duplicate_is_error(semantic_workspace):
    yaml_path = prepare_cracks(semantic_workspace)
    root = paths.dataset_processed_dir("crackmap")
    write_png(root / "images/val/val.png", read_bgr(root / "images/train/train.png"))
    report = validate_semantic_dataset(yaml_path, write_report=False)
    assert next(r for r in report.results if r.name == "mask_split_uniqueness").severity == "ERROR"


def test_invalid_loss_weights_are_rejected():
    with pytest.raises(ValueError, match="必须等于 1"):
        MixerCSegConfig(BCELoss_ratio=0.8, DiceLoss_ratio=0.8)


def test_dataset_error_blocks_model_construction(semantic_workspace, monkeypatch):
    from od_platform.backends.mixercseg import model as model_mod
    from od_platform.training.mixercseg_service import train_mixercseg
    yaml_path = prepare_cracks(semantic_workspace)
    image = paths.dataset_processed_dir("crackmap") / "images/train/train.png"
    image.write_bytes(b"corrupted image")
    def forbidden(*args, **kwargs): raise AssertionError("坏数据不应进入模型构造")
    monkeypatch.setattr(model_mod, "build_model", forbidden)
    with RunContext("training") as run:
        result = train_mixercseg(MixerCSegConfig(epochs=1), yaml_path, run, archive=False)
    assert not result.success and "质检 ERROR" in result.message
    audit = json.loads((result.save_dir / "odp_audit.json").read_text(encoding="utf-8"))
    assert audit["success"] is False


def test_training_audit_marks_smoke_and_selects_only_validation(semantic_workspace, monkeypatch):
    pytest.importorskip("mmengine")
    import torch
    from od_platform.backends.mixercseg import model as model_mod
    from od_platform.training.mixercseg_service import train_mixercseg
    yaml_path = prepare_cracks(semantic_workspace)
    def fake_model(config, checkpoint=None):
        return torch.nn.Conv2d(3, 1, 1), torch.nn.BCEWithLogitsLoss(), "cpu"
    monkeypatch.setattr(model_mod, "build_model", fake_model)
    config = MixerCSegConfig(epochs=1, num_threads=0, max_train_batches=1, max_eval_batches=1)
    with RunContext("training") as run:
        result = train_mixercseg(config, yaml_path, run, archive=False)
    assert result.success, result.message
    audit = json.loads((result.save_dir / "odp_audit.json").read_text(encoding="utf-8"))
    assert audit["smoke"] is True and audit["model_selection_split"] == "val"
    assert audit["history"][0]["train_samples"] == 1
    assert result.best_weight.exists()


def test_training_cancellation_preserves_audit_and_progress(semantic_workspace, monkeypatch):
    pytest.importorskip("mmengine")
    import torch
    from od_platform.backends.mixercseg import model as model_mod
    from od_platform.training.mixercseg_service import train_mixercseg
    yaml_path = prepare_cracks(semantic_workspace)
    monkeypatch.setattr(model_mod, "build_model", lambda *a, **kw: (torch.nn.Conv2d(3, 1, 1), torch.nn.BCEWithLogitsLoss(), "cpu"))
    events = []
    with RunContext("training") as run:
        result = train_mixercseg(MixerCSegConfig(epochs=2, num_threads=0), yaml_path, run,
                                archive=False, on_progress=events.append, should_cancel=lambda: True)
    assert not result.success
    audit = json.loads((result.save_dir / "odp_audit.json").read_text(encoding="utf-8"))
    assert audit["cancelled"] is True and audit["history"] == []
    assert events[0]["phase"] == "validating"
    assert (result.save_dir / "progress.json").exists()


def test_acceptance_uses_only_test_and_reports_unset_quality_target(semantic_workspace, monkeypatch):
    import torch
    from od_platform.backends.mixercseg import model as model_mod
    from od_platform.training.acceptance import evaluate_checkpoint
    yaml_path = prepare_cracks(semantic_workspace)
    calls = []
    class TinyModel(torch.nn.Module):
        def forward(self, inputs):
            calls.append(len(inputs))
            return torch.ones((len(inputs), 1, 512, 512))
    monkeypatch.setattr(model_mod, "build_model", lambda *a, **kw: (TinyModel(), None, "cpu"))
    checkpoint = semantic_workspace.parent / "best.pth"; checkpoint.write_bytes(b"test fixture")
    output = semantic_workspace.parent / "evaluation"
    report = evaluate_checkpoint(MixerCSegConfig(), yaml_path, checkpoint, output)
    assert report["split"] == "test" and report["images"] == 1 and calls == [1]
    assert report["quality_status"] == "needs_review"
    assert report["confusion"] == {"tp": 32, "fp": 224, "fn": 0, "tn": 0}
    assert report["metrics"]["F1"] > report["empty_baseline"]["F1"]
    assert len(list((output / "comparisons").glob("*.png"))) == 1
    assert (output / "per_image.csv").exists()
