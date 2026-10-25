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
