"""Gradio 只提交 API 请求,页面构建不会加载模型或启动训练。"""
import pytest


def test_demo_build_does_not_submit_any_job(monkeypatch):
    pytest.importorskip("gradio")
    from od_web import gradio_demo
    monkeypatch.setattr(gradio_demo, "request", lambda *args, **kwargs: pytest.fail("构建 UI 不应访问或提交任务"))
    demo = gradio_demo.build()
    assert len(demo.blocks) >= 15


def test_demo_submits_video_to_shared_remote_backend(tmp_path, monkeypatch):
    from od_web import gradio_demo
    file = tmp_path / "sample.mp4"; file.write_bytes(b"API client test")
    calls = []
    def request(path, **kwargs):
        calls.append((path, kwargs)); return {"id": "a"*32 + ".mp4"} if "uploads" in path else {"id": "b"*32}
    monkeypatch.setattr(gradio_demo, "request", request)
    identifier, _ = gradio_demo.submit(str(file), "trained:" + "c"*32, 0.25)
    assert identifier == "b"*32 and calls[0][0] == "/api/uploads/video"
    assert calls[1][1]["document"]["media_type"] == "video"


def test_demo_uploads_detection_archive(tmp_path, monkeypatch):
    from od_web import gradio_demo
    file = tmp_path / "dataset.zip"; file.write_bytes(b"archive")
    calls = []
    def request(path, **kwargs):
        calls.append((path, kwargs["data"], kwargs["files"]["files"][0], kwargs["files"]["files"][1].read()))
        return {"id": "ds_example", "status": "importing"}
    monkeypatch.setattr(gradio_demo, "request", request)
    identifier, status = gradio_demo.upload_dataset(str(file), " 港区数据 ")
    assert identifier == "ds_example" and status["status"] == "importing"
    assert calls == [("/api/datasets", {"name": "港区数据", "mode": "archive"}, "dataset.zip", b"archive")]


def test_demo_training_uses_remote_gpu(monkeypatch):
    from od_web import gradio_demo
    calls = []
    def request(path, **kwargs):
        calls.append(kwargs["document"]); return {"id": "job"}
    monkeypatch.setattr(gradio_demo, "request", request)
    gradio_demo.train("wsodd", "yolov8s.pt", 10, 8, 640)
    assert calls[0]["execution"] == "remote" and calls[0]["config"]["device"] == "cuda"
    assert calls[0]["config"]["data"] == "wsodd" and calls[0]["config"]["epochs"] == 10


def test_demo_local_inference_selection(tmp_path, monkeypatch):
    from od_web import gradio_demo
    file = tmp_path / "image.jpg"; file.write_bytes(b"image")
    calls = []
    def request(path, **kwargs):
        calls.append((path, kwargs)); return {"id": "job"}
    monkeypatch.setattr(gradio_demo, "request", request)
    gradio_demo.submit(str(file), "trained:model", 0.25, "local")
    assert calls[1][1]["document"]["execution"] == "local"
