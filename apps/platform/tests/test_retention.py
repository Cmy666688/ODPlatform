"""保留策略只在隔离目录测试,保护引用与未知运行,验证可恢复性。"""
import json
import os
import time

import pytest

from od_platform.common import paths, retention


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for name, value in {"ROOT_DIR": tmp_path, "RUNS_DIR": tmp_path / "runs", "META_DIR": tmp_path / ".odp-meta"}.items(): monkeypatch.setattr(paths, name, value)
    return tmp_path


def old_run(name, *, audited=True):
    directory = paths.RUNS_DIR / "training" / name; directory.mkdir(parents=True)
    (directory / "weights.pt").write_bytes(b"keep my model")
    if audited: (directory / "odp_audit.json").write_text("{}")
    os.utime(directory, (time.time()-86400*90,)*2)
    return directory


def test_default_disabled_and_preview_preserves_data(workspace):
    older = old_run("old"); old_run("recent")
    settings = retention.Policy(keep_latest=1)
    assert retention.preview(settings)["candidates"][0]["path"] == "training/old"
    result = retention.sweep(force=True)
    assert result["disabled"] and older.exists()


def test_references_and_incomplete_audits_are_protected(workspace):
    old_run("referenced"); old_run("unknown", audited=False); old_run("latest")
    directory = paths.RUNS_DIR / "web_jobs" / ("a"*32); directory.mkdir(parents=True)
    (directory / "status.json").write_text(json.dumps({"run_dir": "runs/training/referenced"}))
    plan = retention.preview(retention.Policy(keep_latest=1))
    assert plan["candidates"] == []
    assert len(plan["protected"]) == 3


def test_active_cli_run_is_protected_even_when_old(workspace):
    directory = old_run("active")
    (directory / ".active.json").write_text(json.dumps({"pid": os.getpid()}))
    os.utime(directory, (time.time()-86400*90,)*2)
    old_run("latest")
    plan = retention.preview(retention.Policy(keep_latest=1))
    assert not plan["candidates"]
    assert any(row["protected"] == "活动进程保护" for row in plan["protected"])


def test_enabled_policy_moves_then_restores_exact_content(workspace):
    original = old_run("old"); old_run("newest")
    retention.configure({"enabled": True, "keep_latest": 1})
    result = retention.sweep(force=True)
    assert len(result["moved"]) == 1 and not original.exists()
    record = retention.trash()[0]
    assert (paths.META_DIR / "run_trash" / record["id"] / "payload/weights.pt").read_bytes() == b"keep my model"
    retention.restore(record["id"])
    assert original.is_dir() and (original / "weights.pt").read_bytes() == b"keep my model"
    assert not retention.trash()


def test_expired_quarantine_purges_only_checked_tree(workspace):
    old_run("old"); old_run("newest")
    retention.configure({"enabled": True, "keep_latest": 1})
    retention.sweep(force=True); record = retention.trash()[0]
    file = paths.META_DIR / "run_trash" / record["id"] / "record.json"
    record["moved_at"] = time.time()-86400*20; file.write_text(json.dumps(record))
    outside = workspace / "unrelated.txt"; outside.write_text("preserve")
    assert retention.sweep(force=True)["purged"] == [record["id"]]
    assert outside.read_text() == "preserve"


def test_rejects_path_escape_and_invalid_configuration(workspace):
    with pytest.raises(ValueError): retention.safe_tree(workspace, paths.RUNS_DIR)
    with pytest.raises(ValueError): retention.restore("../outside")
    with pytest.raises(ValueError): retention.configure({"keep_days": 0})
    with pytest.raises(ValueError): retention.configure({"enabled": "true"})
