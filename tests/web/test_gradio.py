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
