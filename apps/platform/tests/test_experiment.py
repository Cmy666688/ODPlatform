"""完整实验必须经过无批次的测试准备阶段,然后完成独立验收。"""
import json
from types import SimpleNamespace

import pytest

from od_platform.common import paths
from od_platform.common.json_io import write_json
from od_platform.training.experiment import run_experiment


@pytest.mark.parametrize("test_fails", [False, True])
def test_full_training_enters_testing_before_first_test_batch(tmp_path, monkeypatch, test_fails):
    monkeypatch.setattr(paths, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr("od_platform.training.experiment.refs.resolve_dataset_yaml", lambda _: tmp_path / "data.yaml")
    job = tmp_path / "job"; job.mkdir()
    write_json(job / "request.json", {"config": {"epochs": 50}, "created_at": "2026-10-26T00:00:00Z"})
    history = [{"epoch": epoch} for epoch in range(1, 51)]
    def train(config, data_yaml, run, **kwargs):
        write_json(run.artifact_path("odp_audit.json"), {"smoke": False, "history": history})
        kwargs["on_progress"]({"phase": "epoch_complete", "epoch": 50, "history": history})
        return SimpleNamespace(success=True, best_weight=tmp_path / "best.pth")
    def evaluate(config, data_yaml, checkpoint, output, **kwargs):
        initial = json.loads((job / "status.json").read_text(encoding="utf-8"))
        assert initial["phase"] == "testing" and initial["progress"] == 0.94
        kwargs["on_progress"]({"phase": "testing", "batch": 1, "batches": 24})
        partial = json.loads((job / "status.json").read_text(encoding="utf-8"))
        assert 0.94 < partial["progress"] < 1
        if test_fails:
            raise RuntimeError("测试阶段故障")
        return {"metrics": {"F1": 0.7}, "quality_status": "needs_review"}
    monkeypatch.setattr("od_platform.training.mixercseg_service.train_mixercseg", train)
    monkeypatch.setattr("od_platform.training.acceptance.evaluate_checkpoint", evaluate)
    assert run_experiment(job) == int(test_fails)
    final = json.loads((job / "status.json").read_text(encoding="utf-8"))
    assert len(final["history"]) == 50 and final["eta_seconds"] == 0
    if test_fails:
        assert final["status"] == "failed" and final["error"] == "测试阶段故障"
    else:
        assert final["status"] == "completed" and final["progress"] == 1
        assert final["metrics"]["F1"] == 0.7
