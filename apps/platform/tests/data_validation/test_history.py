"""历史报告保留基线,只按检查项报告新增和修复。"""
import json

from od_platform.common import paths
from od_platform.data_validation import history


def test_compare_introduced_resolved_changed_and_removed():
    before = {"results": [{"name": "boxes", "severity": "ERROR"}, {"name": "decode", "severity": "PASS"},
                           {"name": "groups", "severity": "WARNING", "message": "old"}, {"name": "retired", "severity": "PASS"}]}
    after = {"results": [{"name": "boxes", "severity": "PASS"}, {"name": "decode", "severity": "ERROR"},
                          {"name": "groups", "severity": "WARNING", "message": "new"}]}
    assert history.compare(after, before) == {"introduced": ["decode"], "resolved": ["boxes"], "changed": ["groups"], "removed_checks": ["retired"]}


def test_snapshots_do_not_overwrite_and_import_legacy(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS_DIR", tmp_path)
    directory = tmp_path / "data_validation/wsodd"; directory.mkdir(parents=True)
    original = {"results": [{"name": "boxes", "severity": "ERROR"}]}
    (directory / "report.json").write_text(json.dumps(original))
    saved = history.save_report("wsodd", {"results": [{"name": "boxes", "severity": "PASS"}]})
    assert saved["comparison"]["resolved"] == ["boxes"]
    assert saved["comparison"]["baseline"].startswith("legacy_")
    next_report = history.save_report("wsodd", {"results": []})
    assert next_report["comparison"]["baseline"] == saved["report_id"]
    assert len(history.listing("wsodd")) == 3
    assert history.listing("wsodd")[0]["report_id"] == next_report["report_id"]
    assert json.loads((directory / "report.json").read_text())["report_id"] == next_report["report_id"]


def test_progress_has_terminal_and_failure_states(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "META_DIR", tmp_path)
    history.progress("wsodd", 10, 20)
    file = history.progress_path("wsodd")
    assert json.loads(file.read_text())["completed"] == 10
    history.progress("wsodd", 20, 20, phase="completed")
    assert json.loads(file.read_text())["status"] == "completed"
    history.progress("wsodd", 10, 20, error="bad image")
    assert json.loads(file.read_text())["status"] == "failed"
