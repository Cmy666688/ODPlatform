"""状态原子写入不能因 Windows 短暂读句柄占用而中断长任务。"""
import json
import os

import pytest

from od_platform.common import json_io


def test_atomic_status_write_retries_sharing_conflict(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    json_io.write_json(path, {"epoch": 1})
    original = os.replace
    attempts = []
    def replace(source, destination):
        attempts.append(1)
        assert json.loads(path.read_text())["epoch"] == 1
        if len(attempts) < 3:
            raise PermissionError("reader holds file")
        original(source, destination)
    monkeypatch.setattr(json_io.os, "replace", replace)
    monkeypatch.setattr(json_io.time, "sleep", lambda _: None)
    json_io.write_json(path, {"epoch": 2})
    assert len(attempts) == 3
    assert json.loads(path.read_text())["epoch"] == 2
    assert len(list(tmp_path.iterdir())) == 1


def test_failed_replace_keeps_previous_status_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    json_io.write_json(path, {"epoch": 1})
    def denied(*args): raise PermissionError("persistent lock")
    monkeypatch.setattr(json_io.os, "replace", denied)
    monkeypatch.setattr(json_io.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        json_io.write_json(path, {"epoch": 2})
    assert json.loads(path.read_text())["epoch"] == 1
    assert len(list(tmp_path.iterdir())) == 1
