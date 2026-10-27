"""Streamlit 页面不能在重跑时隐式提交训练。"""
from pathlib import Path

import httpx


def test_workbench_rerun_without_gpu_never_posts_training(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    calls = []
    def response(path):
        if path == "/api/overview":
            return {"hardware": {"cuda_available": False, "device": "CPU"},
                    "quality": {"results": [{"name": "mask_pair_existence", "severity": "PASS", "summary": "配对正确"}]}}
        return []
    class FakeClient:
        def get(self, path):
            return httpx.Response(200, json=response(path), request=httpx.Request("GET", "http://localhost" + path))
        def post(self, *args, **kwargs):
            calls.append(args); raise AssertionError("页面重跑不应启动训练")
    st.cache_resource.clear()
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: FakeClient())
    app_path = Path(__file__).resolve().parents[2] / "apps/streamlit/app.py"
    app = AppTest.from_file(str(app_path), default_timeout=20).run()
    assert not app.exception
    assert app.button[0].disabled
    app.run()
    assert not app.exception and calls == []
    app.segmented_control[0].set_value("数据质量").run()
    assert not app.exception
    assert len(app.dataframe) == 1
    st.cache_resource.clear()



def test_workbench_switches_monitor_quality_and_results_together(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    datasets = [{"id": key, "name": name, "trainable": True} for key, name in [("wsodd","A 数据"),("ds_other","B 数据")]]
    def response(path):
        if path.startswith("/api/overview"):
            key = "ds_other" if "dataset=ds_other" in path else "wsodd"
            return {"hardware": {"cuda_available": False, "device": "等待远程 GPU"}, "dataset": {"id":key}, "datasets":datasets,
                    "quality":{"results":[{"name":key,"severity":"PASS","summary":"selected " + key}]}}
        if path == "/api/jobs":
            return [{"id":str(index)*32,"name":row["name"]+"实验","kind":"training","status":"completed","history":[],"phase":"completed","progress":1,
                     "config":{"data":row["id"],"epochs":1}} for index,row in enumerate(datasets,1)]
        return []
    class FakeClient:
        def get(self, path): return httpx.Response(200,json=response(path),request=httpx.Request("GET","http://localhost"+path))
        def post(self, *args, **kwargs): raise AssertionError("切换数据不启动训练")
    st.cache_resource.clear(); monkeypatch.setattr(httpx,"Client",lambda **kwargs: FakeClient())
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2]/"apps/streamlit/app.py"),default_timeout=20).run()
    app.sidebar.selectbox[0].select("ds_other").run()
    assert not app.exception
    assert any(box.options == ["B 数据实验"] for box in app.selectbox)
    app.segmented_control[0].set_value("数据质量").run()
    assert not app.exception and app.dataframe[0].value.iloc[0]["检查项"] == "ds_other"
    app.segmented_control[0].set_value("结果浏览").run()
    assert not app.exception and any(box.options == ["B 数据实验"] for box in app.selectbox)
    st.cache_resource.clear()


def test_workbench_offers_local_cpu_when_remote_unavailable(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    dataset = {"id": "wsodd", "name": "WSODD", "trainable": True}
    def response(path):
        if path.startswith("/api/overview"):
            return {"hardware": {"cuda_available": False, "device": "未连接"}, "dataset": dataset, "datasets": [dataset]}
        if path == "/api/jobs":
            return [{"id": "1" * 32, "name": "检测权重", "kind": "training", "status": "completed", "history": [],
                     "phase": "completed", "progress": 1, "config": {"data": "wsodd", "backend": "yolo", "epochs": 1}}]
        return []
    class FakeClient:
        def get(self, path): return httpx.Response(200, json=response(path), request=httpx.Request("GET", "http://localhost" + path))
        def post(self, *args, **kwargs): raise AssertionError("浏览页面不能自动提交任务")
    st.cache_resource.clear(); monkeypatch.setattr(httpx, "Client", lambda **kwargs: FakeClient())
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "apps/streamlit/app.py"), default_timeout=20).run()
    app.segmented_control[0].set_value("图片与视频检测").run()
    assert not app.exception
    assert any(box.label == "推理设备" and box.value == "本机 CPU" for box in app.selectbox)
    assert any(button.label == "开始检测" and not button.disabled for button in app.button)
    st.cache_resource.clear()


def test_workbench_routes_uploaded_video_to_video_endpoint(monkeypatch):
    from io import BytesIO
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    dataset = {"id": "wsodd", "name": "WSODD", "trainable": True}
    calls = []
    upload = BytesIO(b"video"); upload.name = "sample.mp4"; upload.type = "video/mp4"
    def response(path):
        if path.startswith("/api/overview"):
            return {"hardware": {"cuda_available": False, "device": "未连接"}, "dataset": dataset, "datasets": [dataset]}
        if path == "/api/jobs":
            return [{"id": "1" * 32, "name": "检测权重", "kind": "training", "status": "completed", "history": [],
                     "phase": "completed", "progress": 1, "config": {"data": "wsodd", "backend": "yolo", "epochs": 1}}]
        return {"status": "running", "phase": "inference", "progress": 0}
    class FakeClient:
        cookies = {"csrftoken": "token"}
        def get(self, path, **kwargs): return httpx.Response(200, json=response(path), request=httpx.Request("GET", "http://localhost" + path))
        def post(self, path, **kwargs):
            calls.append((path, kwargs))
            return httpx.Response(202, json={"id": "job"}, request=httpx.Request("POST", "http://localhost" + path))
    st.cache_resource.clear(); monkeypatch.setattr(httpx, "Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "apps/streamlit/app.py"), default_timeout=20).run()
    app.segmented_control[0].set_value("图片与视频检测").run()
    next(button for button in app.button if button.label == "开始检测").click().run()
    assert not app.exception
    assert [call[0] for call in calls] == ["/api/uploads/video", "/api/jobs"]
    assert calls[1][1]["json"]["media_type"] == "video"
    assert calls[1][1]["json"]["execution"] == "local"
    st.cache_resource.clear()


def test_workbench_shows_review_limits_and_distinct_metrics(monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest
    dataset = {"id": "wsodd", "name": "WSODD", "trainable": True}
    def response(path):
        if path.startswith("/api/overview"):
            return {"hardware": {"cuda_available": False, "device": "未连接"}, "dataset": dataset, "datasets": [dataset],
                    "quality": {"results": [{"name": "detection_eval_coverage", "severity": "WARNING", "summary": "验证集缺少 tree"},
                                            {"name": "detection_split_policy", "severity": "WARNING", "summary": "分组来源待确认"}]}}
        if path == "/api/jobs":
            return [{"id": "1" * 32, "name": "实验", "kind": "training", "status": "completed", "phase": "completed", "progress": 1,
                     "evaluation_independent": False, "metrics": {"mAP50": .747}, "config": {"data": "wsodd", "epochs": 1},
                     "history": [{"epoch": 1, "train_loss": 1.0, "validation": {"loss": 2.0, "mAP50_95": .446}}]}]
        return []
    class FakeClient:
        def get(self, path): return httpx.Response(200, json=response(path), request=httpx.Request("GET", "http://localhost" + path))
        def post(self, *args, **kwargs): raise AssertionError("页面不能自动提交任务")
    st.cache_resource.clear(); monkeypatch.setattr(httpx, "Client", lambda **kwargs: FakeClient())
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "apps/streamlit/app.py"), default_timeout=20).run()
    assert not app.exception
    assert any(item.label == "验证 mAP50–95" and item.value == "44.6%" for item in app.metric)
    assert any(item.label == "同源对照 mAP50" and item.value == "74.7%" for item in app.metric)
    assert any("采集组重叠" in item.value for item in app.caption)
    app.segmented_control[0].set_value("数据质量").run()
    frame = app.dataframe[0].value
    assert list(frame["等级"]) == ["通过（检查完成）", "通过（检查完成）"]
    assert "缺少 tree" in frame.iloc[0]["说明"] and "待确认" in frame.iloc[1]["说明"]
    st.cache_resource.clear()
