"""视频接入、远程边界与解释。模型结果使用桩,测试不执行 GPU 计算。"""
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from od_platform.common import paths
from od_platform.common.video import inspect_video


def clip(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 5, (64,64))
    assert writer.isOpened()
    for index in range(10): writer.write(np.full((64,64,3), index*20, dtype=np.uint8))
    writer.release(); return path


def test_video_upload_and_gpu_admission_without_local_fallback(web, monkeypatch):
    from od_web import app
    client, store, root = web
    source = clip(root / "harbor.mp4")
    value = client.post("/api/uploads/video", files={"file": (source.name, source.read_bytes(), "video/mp4")})
    assert value.status_code == 200 and value.json()["frames"] == 10
    identifier = "3"*32; folder = store.root / identifier; folder.mkdir()
    weight = root / "best.pt"; weight.write_bytes(b"test weights")
    (folder / "status.json").write_text(json.dumps({"id": identifier, "kind": "training", "status": "completed",
        "config": {"backend": "yolo", "data": "wsodd", "imgsz": 960, "architecture": "yolov8s.pt"}, "best_weight": str(weight)}))
    payload = {"kind": "inference", "media_type": "video", "upload_id": value.json()["id"], "model_id": "trained:" + identifier}
    assert client.post("/api/jobs", json=payload).status_code == 409
    monkeypatch.setattr(app, "_hardware", lambda: {"cuda_available": True})
    documents = []; monkeypatch.setattr(store, "create", lambda row: documents.append(row) or {"id": "b"*32})
    assert client.post("/api/jobs", json=payload).status_code == 202
    assert documents[0]["config"]["device"] == "cuda" and documents[0]["media_type"] == "video"
    assert documents[0]["config"]["imgsz"] == 960 and documents[0]["config"]["architecture"] == "yolov8s.pt"
    assert client.post("/api/jobs", json={**payload,"upload_id": "../" + value.json()["id"]}).status_code == 422


def test_explicit_local_cpu_inference_does_not_require_remote_gpu(web, monkeypatch):
    from od_web import app
    client, store, root = web
    source = clip(root / "local.mp4")
    upload = client.post("/api/uploads/video", files={"file": (source.name, source.read_bytes(), "video/mp4")}).json()
    identifier = "5" * 32; folder = store.root / identifier; folder.mkdir()
    weight = root / "best.pt"; weight.write_bytes(b"test weights")
    (folder / "status.json").write_text(json.dumps({"id": identifier, "kind": "training", "status": "completed",
        "config": {"backend": "yolo", "data": "wsodd", "imgsz": 960, "architecture": "yolov8s.pt"}, "best_weight": str(weight)}))
    monkeypatch.setattr(app, "_hardware", lambda: pytest.fail("CPU 推理不应检查远程 GPU"))
    documents = []; monkeypatch.setattr(store, "create", lambda row: documents.append(row) or {"id": "b" * 32})
    payload = {"kind": "inference", "media_type": "video", "upload_id": upload["id"], "model_id": "trained:" + identifier, "execution": "local"}
    assert client.post("/api/jobs", json=payload).status_code == 202
    assert documents[0]["config"]["device"] == "cpu" and documents[0]["execution"] == "local"
    assert client.post("/api/jobs", json={"kind": "training", "execution": "local"}).status_code == 422


def test_video_rejects_invalid_input_and_protects_csrf(web):
    client, _, root = web
    assert client.post("/api/uploads/video", files={"file": ("bad.mp4", b"invalid")}).status_code == 422
    assert client.post("/api/uploads/video", files={"file": ("bad.exe", b"invalid")}).status_code == 413
    assert client.post("/api/uploads/video", headers={"X-CSRFToken": "wrong"}).status_code == 403
    file = root / "bad.mp4"; file.write_bytes(b"invalid")
    with pytest.raises(ValueError): inspect_video(file)


def test_video_pipeline_outputs_playable_artifacts_and_honest_counts(web, monkeypatch):
    import torch
    from od_platform.runtime_config.detection import DetectionConfig
    from od_platform.training.video_detection import infer_video
    _, _, root = web; source = clip(root / "harbor.mp4"); weight = root / "best.pt"; weight.write_bytes(b"test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    image = np.full((64,64,3), 128, dtype=np.uint8)
    class FakeModel:
        def __init__(self, model): pass
        def predict(self, **kwargs):
            assert kwargs["device"] == 0 and kwargs["stream"] is True
            for _ in range(10):
                yield SimpleNamespace(orig_shape=(64,64), plot=lambda: image,
                    summary=lambda: [{"name": "ship", "class": 0, "confidence": 0.8, "box": {"x1": 1,"x2": 20,"y1": 1,"y2": 20}}])
    state = {}
    assert infer_video(DetectionConfig(model=str(weight)), str(source), state.update, lambda: False, model_factory=FakeModel) == 0
    run = root / state["run_dir"]; summary = json.loads((run / "video_summary.json").read_text(encoding="utf-8"))
    assert state["status"] == "completed" and summary["frames"] == 10
    assert summary["class_summary"]["ship"]["frame_occurrences"] == 10
    assert summary["class_summary"]["ship"]["max_simultaneous"] == 1
    assert len((run / "frames.jsonl").read_text().splitlines()) == 10
    assert inspect_video(run / "annotated.mp4")["frames"] == 10 and not (run / "annotated.raw.mp4").exists()
    def broken_model(model): raise RuntimeError("模型无法加载")
    with pytest.raises(RuntimeError, match="模型无法加载"):
        infer_video(DetectionConfig(model=str(weight)), str(source), state.update, lambda: False, model_factory=broken_model)
    failed_run = root / state["run_dir"]
    assert not (failed_run / "annotated.raw.mp4").exists() and not (failed_run / ".active.json").exists()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"): infer_video(DetectionConfig(model=str(weight)), str(source), state.update, lambda: False, model_factory=FakeModel)
    class CpuModel(FakeModel):
        def predict(self, **kwargs):
            assert kwargs["device"] == "cpu"
            image = np.full((64,64,3), 128, dtype=np.uint8)
            for _ in range(10):
                yield SimpleNamespace(orig_shape=(64,64), plot=lambda: image, summary=lambda: [])
    assert infer_video(DetectionConfig(model=str(weight), device="cpu"), str(source), state.update, lambda: False, model_factory=CpuModel) == 0
    audit = json.loads((root / state["run_dir"] / "odp_audit.json").read_text(encoding="utf-8"))
    assert audit["resolved_device"] == "cpu"


def test_video_range_download_and_explanation_use_summary_only(web, monkeypatch):
    from od_web import assistant, explanation
    client, store, root = web; identifier = "4"*32; directory = store.root / identifier; directory.mkdir()
    run = paths.RUNS_DIR / "detection_infer/example"; run.mkdir(parents=True)
    (run / "annotated.mp4").write_bytes(b"0123456789")
    (run / "video_summary.json").write_text(json.dumps({"media_type": "video", "frames": 10, "class_summary": {"ship": {"frame_occurrences": 10}}, "private_path": "must not send"}))
    (directory / "status.json").write_text(json.dumps({"id": identifier, "status": "completed", "kind": "inference", "media_type": "video", "run_dir": "runs/detection_infer/example"}))
    response = client.get(f"/api/jobs/{identifier}/files/run/annotated.mp4", headers={"Range":"bytes=2-5"})
    assert response.status_code == 206 and response.content == b"2345"
    assert response.headers["Content-Range"] == "bytes 2-5/10"
    assert client.get(f"/api/jobs/{identifier}/files/run/annotated.mp4", headers={"Range":"bytes=100-"}).status_code == 416
    assert client.get(f"/api/jobs/{identifier}/files/run/annotated.mp4?download=true").content == b"0123456789"
    # ASGI 使用原生异步迭代器,避免 Django 将整个视频读入内存再发送。
    import asyncio
    from django.test import AsyncRequestFactory
    from od_web import views
    async def collect(range_header):
        request = AsyncRequestFactory().get("/video", headers={"Range": range_header} if range_header else {})
        request.META["od.job_store"] = store
        response = views.artifact(request, identifier, "run/annotated.mp4")
        assert response.is_async
        return b"".join([chunk async for chunk in response.streaming_content])
    assert asyncio.run(collect("bytes=2-5")) == b"2345"
    assert asyncio.run(collect(None)) == b"0123456789"
    def ask(message, evidence):
        assert "must not send" not in json.dumps(evidence)
        assert evidence["video_analysis"]["class_summary"]["ship"]["frame_occurrences"] == 10
        return {"answer": "共 10 帧出现船舶，不等于 10 艘。", "provider":"DeepSeek", "model":"test", "usage":{}}
    monkeypatch.setattr(assistant, "ask", ask)
    (directory / "explanation.json").write_text('{"status":"running"}')
    result = explanation.run(directory)
    assert result["status"] == "completed"
    assert client.get(f"/api/jobs/{identifier}/explanation").json()["answer"] == result["answer"]
    # 重试时,子进程即使在 Popen 返回前已完成,父进程也不能覆盖回答为 running。
    (directory / "explanation.json").write_text('{"status":"failed"}')
    monkeypatch.setattr(assistant, "status", lambda: {"configured": True})
    def instant_process(*args, **kwargs):
        import os
        explanation.run(directory)
        return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(explanation.subprocess, "Popen", instant_process)
    assert explanation.start(store, identifier)["status"] == "completed"
