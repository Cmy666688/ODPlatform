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
