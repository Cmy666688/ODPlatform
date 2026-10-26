"""远程任务边界:禁止 CPU 回退、传输身份、重连与历史状态。"""
import json
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from od_platform.common import paths
from od_platform.common.json_io import write_json
from od_web import remote, remote_agent

from test_datasets import upload_template


@pytest.fixture
def inline_import(monkeypatch):
    from od_web import datasets
    from od_web.dataset_worker import run
    monkeypatch.setattr(datasets, "launch_import", lambda record, source, mode: run(record.id, source, mode))


def test_remote_offline_blocks_cpu_draft_and_inference_without_local_fallback(web, inline_import, monkeypatch):
    client, store, root = web
    identifier = upload_template(client).json()["id"]
    monkeypatch.setattr("od_web.app._hardware", lambda: {"cuda_available": False, "message": "远程 GPU 不可用"})
    monkeypatch.setattr(store, "create", lambda _: pytest.fail("断线时不能启动本机工作进程"))
    response = client.post("/api/jobs", json={"kind": "training", "require_cuda": False,
                           "config": {"data": identifier, "device": "cpu", "scan_backend": "cpu_jit"}})
    assert response.status_code == 409 and "远程 GPU" in response.json()["detail"]
    weight = root / "models/pretrained/checkpoint_CrackMap.pth"; weight.parent.mkdir(parents=True); weight.touch()
    monkeypatch.setattr(paths, "PRETRAINED_MODELS_DIR", weight.parent)
    upload = root / "runs/web_uploads" / ("e" * 32 + ".png"); upload.parent.mkdir(); upload.touch()
    response = client.post("/api/jobs", json={"kind": "inference", "upload_id": upload.name})
    assert response.status_code == 409


def test_remote_inference_config_never_depends_on_local_torch(web, monkeypatch):
    client, store, root = web
    weight = root / "models/pretrained/checkpoint_CrackMap.pth"; weight.parent.mkdir(parents=True); weight.touch()
    monkeypatch.setattr(paths, "PRETRAINED_MODELS_DIR", weight.parent)
    upload = root / "runs/web_uploads" / ("e" * 32 + ".png"); upload.parent.mkdir(); upload.touch()
    monkeypatch.setattr("od_web.app._hardware", lambda: {"cuda_available": True})
    requests = []
    monkeypatch.setattr(store, "create", lambda document: requests.append(document) or {"id": "a" * 32})
    response = client.post("/api/jobs", json={"kind": "inference", "upload_id": upload.name, "config": {"device": "cpu"}})
    assert response.status_code == 202
    assert requests[0]["config"]["device"] == "cuda" and requests[0]["config"]["scan_backend"] == "reference"


def test_prepare_transfers_dataset_manifest_and_every_pair(web, inline_import):
    client, store, root = web
    identifier = upload_template(client).json()["id"]
    directory = store.root / ("f" * 32); directory.mkdir()
    write_json(directory / "status.json", {"status": "queued", "execution": "remote", "remote_managed": True})
    files = []; documents = {}
    class Transfer:
        config = SimpleNamespace(root="/gpu/project")
        sftp = SimpleNamespace(stat=lambda _: (_ for _ in ()).throw(FileNotFoundError()))
        def put(self, source, target): files.append((source, str(target)))
        def put_json(self, value, target): documents[str(target)] = value
    remote.prepare(Transfer(), directory, {"kind": "training", "config": {"data": identifier, "device": "cpu"}})
    assert len([target for _, target in files if target.endswith(".png")]) == 18
    assert any(target.endswith("semantic_manifest.json") for _, target in files)
    assert any(target.endswith(identifier + ".yaml") for _, target in files)
    request = documents[f"/gpu/project/runs/web_jobs/{directory.name}/request.json"]
    assert request["config"]["device"] == "cuda"
    initial = documents[f"/gpu/project/runs/web_jobs/{directory.name}/status.json"]
    assert initial["execution"] == "local" and "remote_managed" not in initial


@pytest.mark.parametrize("value", ["../secret", "/root/secret", "C:/secret", "runs/../../secret", "runs\\secret", "models/secret"])
def test_download_directory_cannot_escape_runs(value):
    with pytest.raises(ValueError): remote.relative_file(value, "runs")


def test_reconnect_reuses_existing_launch_marker_without_spawning(web, monkeypatch):
    _, store, _ = web
    directory = store.root / ("d" * 32); directory.mkdir()
    write_json(directory / "status.json", {"id": directory.name, "status": "completed", "created_at": "2026-10-26"})
    write_json(directory / "launch.json", {"pid": 12345})
    monkeypatch.setattr(remote_agent, "job_lock", __import__("contextlib").nullcontext)
    monkeypatch.setattr(remote_agent.subprocess, "Popen", lambda *a, **k: pytest.fail("重连不能重跑任务"))
    assert remote_agent.launch(directory.name)["pid"] == 12345


def test_bridge_recovers_ambiguous_launch_and_delivers_pending_cancel(web, monkeypatch):
    _, store, _ = web
    directory = store.root / ("a" * 32); directory.mkdir()
    write_json(directory / "request.json", {"kind": "inference", "config": {}})
    write_json(directory / "status.json", {"id": directory.name, "status": "queued", "config": {}})
    (directory / "cancel.request").touch()
    launches = []; cancels = []
    class Transfer:
        def __init__(self, _): self.sftp = SimpleNamespace(open=self.missing)
        def missing(self, *args): raise FileNotFoundError()
        def close(self): pass
        def put_json(self, value, target): cancels.append(str(target))
        def get_json(self, target): raise FileNotFoundError()
        def command(self, arguments):
            if arguments[0] == "launch":
                launches.append(arguments[1])
                if len(launches) == 1: raise TimeoutError("确认启动前断线")
                return {"pid": 44}
            return {"id": directory.name, "status": "cancelled", "config": {"device": "cuda"}}
    monkeypatch.setattr(remote.RemoteConfig, "load", lambda: SimpleNamespace(root="/gpu", host="test"))
    monkeypatch.setattr(remote, "Connection", Transfer)
    monkeypatch.setattr(remote, "prepare", lambda *args: PurePosixPath("/gpu/runs/web_jobs") / directory.name)
    monkeypatch.setattr(remote.time, "sleep", lambda _: None)
    remote.bridge(directory)
    final = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    assert final["status"] == "cancelled" and final["execution"] == "remote"
    assert len(launches) == 2 and len(cancels) == 2 and final["artifacts_synced_at"]


def test_expired_hardware_cache_does_not_report_stale_gpu_ready(web, monkeypatch):
    def offline(): raise OSError("offline")
    remote._CACHE.update(key=str(paths.META_DIR), time=0, value={"cuda_available": True})
    monkeypatch.setattr(remote, "Connection", offline)
    result = remote.hardware(force=True)
    assert not result["cuda_available"] and not result["connected"] and "offline" in result["message"]


def test_recovered_bridge_does_not_overwrite_running_inputs_or_relaunch(web, monkeypatch):
    _, store, _ = web
    directory = store.root / ("9" * 32); directory.mkdir()
    write_json(directory / "request.json", {"kind": "inference", "config": {}})
    write_json(directory / "status.json", {"status": "running"})
    class Transfer:
        def __init__(self, _): self.sftp = SimpleNamespace(open=self.missing)
        def missing(self, *args): raise FileNotFoundError()
        def close(self): pass
        def get_json(self, target):
            if target.name == "launch.json": return {"pid": 44}
            raise FileNotFoundError()
        def command(self, arguments):
            assert arguments[0] == "snapshot"
            return {"status": "completed", "config": {"device": "cuda"}}
    monkeypatch.setattr(remote.RemoteConfig, "load", lambda: SimpleNamespace(root="/gpu", host="test"))
    monkeypatch.setattr(remote, "Connection", Transfer)
    monkeypatch.setattr(remote, "prepare", lambda *a: pytest.fail("恢复时不能覆盖远端运行数据"))
    remote.bridge(directory)
    assert json.loads((directory / "status.json").read_text(encoding="utf-8"))["status"] == "completed"


def test_listing_recovers_dead_bridge_but_keeps_finished_history(web, monkeypatch):
    client, store, _ = web
    starts = []
    monkeypatch.setattr(remote, "start_bridge", lambda directory: starts.append(directory.name))
    monkeypatch.setattr("psutil.pid_exists", lambda _: False)
    for identifier, status in [("7" * 32, "running"), ("8" * 32, "completed")]:
        directory = store.root / identifier; directory.mkdir()
        write_json(directory / "status.json", {"id": identifier, "created_at": "2026-10-26T00:00:00+00:00",
                   "status": status, "execution": "remote", "remote_managed": True})
        write_json(directory / "bridge.json", {"pid": 987654321})
    assert client.get("/api/jobs").status_code == 200
    assert starts == ["7" * 32]


def test_worker_dependency_failure_is_reported_with_original_cause(web, monkeypatch):
    _, store, _ = web
    directory = store.root / ("6" * 32); directory.mkdir()
    write_json(directory / "status.json", {"id": directory.name, "status": "queued"})
    def fail(_): raise ImportError("missing inference dependency")
    monkeypatch.setattr("od_web.worker.run", fail)
    monkeypatch.setattr(remote_agent, "hardware", lambda: {"cuda_available": True})
    assert remote_agent.run(directory.name) == 1
    final = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    assert final["status"] == "failed" and final["error"] == "missing inference dependency"
