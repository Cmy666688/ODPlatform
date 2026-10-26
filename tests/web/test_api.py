"""Web 边界:无 GPU 不启动训练,上传校验、任务状态和产物访问。"""
from __future__ import annotations

import os

import numpy as np
import pytest

from od_platform.common import paths
from od_platform.common.json_io import write_json
from od_platform.common.semantic import write_png


def test_without_gpu_complete_training_never_starts(web, monkeypatch):
    client, store, _ = web
    def forbidden(*args, **kwargs): raise AssertionError("不应启动训练进程")
    monkeypatch.setattr(store, "create", forbidden)
    response = client.post("/api/jobs", json={"kind": "training"})
    assert response.status_code == 409
    assert client.get("/api/jobs").json() == []


def test_incomplete_smoke_configuration_cannot_be_submitted_as_full(web):
    response = web[0].post("/api/jobs", json={"kind": "training", "config": {"max_train_batches": 1}})
    assert response.status_code == 422


def test_upload_is_materialized_under_opaque_id_and_bad_image_is_rejected(web):
    client, _, tmp_path = web
    image = tmp_path / "input.png"; write_png(image, np.zeros((10, 14, 3), dtype=np.uint8))
    response = client.post("/api/uploads", files={"file": ("../../input.png", image.read_bytes(), "image/png")})
    assert response.status_code == 200
    data = response.json()
    assert data["width"] == 14 and data["height"] == 10
    assert (paths.RUNS_DIR / "web_uploads" / data["id"]).is_file()
    assert client.post("/api/uploads", files={"file": ("bad.png", b"not image")}).status_code == 422


def test_arbitrary_upload_path_and_unknown_model_are_rejected(web):
    client, _, _ = web
    assert client.post("/api/jobs", json={"kind": "inference", "upload_id": "../secret.png"}).status_code == 422


def test_artifacts_cannot_escape_job_directory(web):
    client, store, tmp_path = web
    job_id = "a" * 32; directory = store.root / job_id; directory.mkdir()
    write_json(directory / "status.json", {"id": job_id, "status": "completed", "created_at": "2026-10-26", "kind": "inference"})
    (tmp_path / "secret.json").write_text("private", encoding="utf-8")
    (directory / "report.json").write_text('{"ok": true}', encoding="utf-8")
    assert client.get(f"/api/jobs/{job_id}/files/report.json").json() == {"ok": True}
    with pytest.raises(FileNotFoundError): store.artifact(job_id, "../../../secret.json")
    assert client.get("/api/jobs/invalid").status_code == 404


def test_cancel_is_persisted_and_finished_task_cannot_be_cancelled(web):
    client, store, _ = web
    job_id = "b" * 32; directory = store.root / job_id; directory.mkdir()
    state = {"id": job_id, "status": "running", "created_at": "2026-10-26", "pid": os.getpid(), "kind": "training"}
    write_json(directory / "status.json", state)
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 200
    assert (directory / "cancel.request").exists()
    assert client.get(f"/api/jobs/{job_id}").json()["cancel_requested"] is True
    write_json(directory / "status.json", {**state, "status": "completed"})
    assert client.post(f"/api/jobs/{job_id}/cancel").status_code == 409


def test_cross_origin_task_submission_is_rejected(web):
    assert web[0].post("/api/jobs", headers={"Origin": "https://unrelated.example"}, json={"kind": "training"}).status_code == 403


def test_frontend_and_schema_are_served(web):
    client, _, _ = web
    assert "训练监控" in client.get("/").text and "数据集管理" in client.get("/").text
    assert client.get("/assets/app.js").status_code == 200
    document = client.get("/api/config").json()
    assert document["defaults"]["epochs"] == 50
    assert document["defaults"]["max_train_batches"] is None
    assert "lr" in document["schema"]["properties"]


@pytest.mark.parametrize("execution,expected", [("remote", "running"), ("local", "failed")])
def test_pid_liveness_uses_execution_host(web, monkeypatch, execution, expected):
    client, store, _ = web
    job_id = "c" * 32; directory = store.root / job_id; directory.mkdir()
    write_json(directory / "status.json", {"id": job_id, "status": "running", "execution": execution,
               "pid": 987654321, "kind": "training", "created_at": "2026-10-26"})
    monkeypatch.setattr("od_web.jobs.psutil.pid_exists", lambda _: False)
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == expected
