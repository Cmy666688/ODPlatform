from pathlib import Path
import sys

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps/web-backend/src"))


@pytest.fixture
def web(tmp_path, monkeypatch, db):
    from od_platform.common import paths, refs
    from od_web import app as app_mod
    from od_web.jobs import JobStore
    for name, value in {"ROOT_DIR": tmp_path, "META_DIR": tmp_path / ".odp-meta", "RUNS_DIR": tmp_path / "runs",
                        "PROCESSED_DATA_DIR": tmp_path / "data/processed", "DATASET_CONFIGS_DIR": tmp_path / "configs/datasets"}.items():
        monkeypatch.setattr(paths, name, value)
    monkeypatch.setattr(refs, "DATASET_CONFIGS_DIR", paths.DATASET_CONFIGS_DIR)
    store = JobStore(paths.RUNS_DIR / "web_jobs")
    monkeypatch.setattr(app_mod, "_hardware", lambda: {"cuda_available": False, "device": "CPU"})
    application = app_mod.create_app(store)
    def wsgi(environ, start_response):
        # HTTPX 提供解码后的 path;PEP 3333 要求 WSGI PATH_INFO 使用 latin-1 包装原始字节。
        environ["PATH_INFO"] = environ["PATH_INFO"].encode("utf-8").decode("latin-1")
        return application(environ, start_response)
    with httpx.Client(transport=httpx.WSGITransport(app=wsgi), base_url="http://testserver") as client:
        client.get("/"); client.headers["X-CSRFToken"] = client.cookies["csrftoken"]
        yield client, store, tmp_path
