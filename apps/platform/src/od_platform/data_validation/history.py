"""质检进度、不可覆盖的历史快照与按检查项比较。"""
from datetime import datetime, timezone
import json
import os
from uuid import uuid4

from od_platform.common import paths
from od_platform.common.json_io import write_json


def progress_path(identifier): return paths.META_DIR / "data_validation" / (identifier + "_progress.json")


def progress(identifier, done, total, *, phase="validating", error=""):
    write_json(progress_path(identifier), {"status": "failed" if error else "completed" if phase == "completed" else "running",
               "phase": phase, "completed": done, "total": total, "error": error, "pid": os.getpid(),
               "updated_at": datetime.now(timezone.utc).isoformat()})


def compare(current, previous):
    before = {row["name"]: row for row in previous.get("results", [])}
    after = {row["name"]: row for row in current.get("results", [])}
    def bad(row): return row.get("severity") in ("ERROR", "WARNING")
    changes = {"introduced": [], "resolved": [], "changed": [], "removed_checks": sorted(before.keys() - after.keys())}
    for name, row in after.items():
        old = before.get(name)
        if bad(row) and (old is None or not bad(old)): changes["introduced"].append(name)
        elif old and bad(old) and not bad(row): changes["resolved"].append(name)
        elif old and row != old: changes["changed"].append(name)
    return changes


def save_report(identifier, report):
    directory = paths.RUNS_DIR / "data_validation" / identifier
    latest = directory / "report.json"; previous = json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else None
    history = directory / "history"; history.mkdir(parents=True, exist_ok=True)
    if previous and not previous.get("report_id"):
        previous["report_id"] = "legacy_" + uuid4().hex[:12]
        write_json(history / (previous["report_id"] + ".json"), previous)
    report = {**report, "report_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ_") + uuid4().hex[:8],
              "checked_at": datetime.now(timezone.utc).isoformat()}
    report["comparison"] = {"baseline": previous.get("report_id") if previous else None,
                            **compare(report, previous or {})}
    write_json(history / (report["report_id"] + ".json"), report); write_json(latest, report)
    return report


def listing(identifier):
    history = paths.RUNS_DIR / "data_validation" / identifier / "history"
    reports = [json.loads(file.read_text(encoding="utf-8")) for file in history.glob("*.json")]
    return sorted(reports, key=lambda report: report.get("checked_at", ""), reverse=True)
