"""运行目录保留策略:预览、引用保护、隔离回收、恢复与延迟清理。"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import shutil
import threading
import time
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
import psutil

from od_platform.common import paths
from od_platform.common.json_io import write_json

BUCKETS = ("training", "detection_infer", "semantic_infer", "inference")
_LOCK = threading.Lock()


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    enabled: bool = False
    keep_days: int = Field(default=30, ge=1, le=3650)
    keep_latest: int = Field(default=3, ge=1, le=1000)
    max_gb: int = Field(default=10, ge=1, le=10000)
    trash_days: int = Field(default=14, ge=7, le=365)


def policy():
    file = paths.META_DIR / "retention/policy.json"
    return Policy(**json.loads(file.read_text(encoding="utf-8"))) if file.exists() else Policy()


def configure(document):
    result = Policy(**document)
    write_json(paths.META_DIR / "retention/policy.json", result.model_dump())
    return result.model_dump()


def safe_tree(file, root):
    root = root.resolve()
    if file.is_symlink() or getattr(file, "is_junction", lambda: False)() or not file.resolve().is_relative_to(root) or file.resolve() == root:
        raise ValueError("运行目录超出允许范围或包含链接")
    for child in file.rglob("*"):
        if child.is_symlink() or getattr(child, "is_junction", lambda: False)(): raise ValueError("运行目录包含链接")


def protected_runs():
    # 所有登记任务引用的目录都保护,包含已结束任务、活动任务和其权重。
    protected = set()
    for file in (paths.RUNS_DIR / "web_jobs").glob("*/status.json"):
        try:
            value = json.loads(file.read_text(encoding="utf-8"))
            if value.get("run_dir"): protected.add((paths.ROOT_DIR / value["run_dir"]).resolve())
        except (ValueError, OSError):
            raise ValueError("任务台账不可读,为保护模型暂不清理") from None
    return protected


def preview(settings=None):
    settings = settings or policy(); protected = protected_runs(); rows = []; total = 0
    for bucket in BUCKETS:
        folders = sorted((file for file in (paths.RUNS_DIR / bucket).glob("*") if file.is_dir()), key=lambda file: file.stat().st_mtime, reverse=True)
        for index, file in enumerate(folders):
            try: safe_tree(file, paths.RUNS_DIR)
            except ValueError: continue
            size = sum(child.stat().st_size for child in file.rglob("*") if child.is_file()); total += size
            reason = "任务引用保护" if file.resolve() in protected else "保留最近运行" if index < settings.keep_latest else ""
            active = file / ".active.json"
            if active.exists():
                try:
                    if psutil.pid_exists(json.loads(active.read_text())["pid"]): reason = "活动进程保护"
                except (ValueError, OSError, KeyError): reason = "活动标记异常,保守保留"
            # 没有审计文件的目录无法证明已经结束,默认保留。
            if not reason and not (file / "odp_audit.json").is_file(): reason = "缺少完成审计,保守保留"
            rows.append({"path": file.relative_to(paths.RUNS_DIR).as_posix(), "bytes": size, "modified": file.stat().st_mtime, "protected": reason})
    remaining = total; selected = []
    for row in sorted(rows, key=lambda item: item["modified"]):
        if not row["protected"] and (time.time() - row["modified"] > settings.keep_days * 86400 or remaining > settings.max_gb * 1024**3):
            selected.append(row); remaining -= row["bytes"]
    return {"policy": settings.model_dump(), "total_bytes": total, "candidates": selected, "protected": [row for row in rows if row["protected"]],
            "reclaimable_bytes": sum(row["bytes"] for row in selected)}


def trash():
    return [json.loads(file.read_text(encoding="utf-8")) for file in (paths.META_DIR / "run_trash").glob("*/record.json")]


def restore(identifier):
    if len(identifier) != 32 or any(c not in "0123456789abcdef" for c in identifier): raise ValueError("非法回收记录")
    folder = paths.META_DIR / "run_trash" / identifier
    safe_tree(folder, paths.META_DIR / "run_trash")
    record = json.loads((folder / "record.json").read_text(encoding="utf-8")); target = paths.RUNS_DIR / record["path"]
    safe_tree(target, paths.RUNS_DIR)
    if target.exists(): raise ValueError("原运行目录已存在,不能覆盖")
    target.parent.mkdir(parents=True, exist_ok=True); (folder / "payload").rename(target)
    (folder / "record.json").unlink(); folder.rmdir()
    return {"restored": record["path"]}


def sweep(*, force=False):
    with _LOCK:
        settings = policy(); plan = preview(settings)
        if not settings.enabled: return {**plan, "moved": [], "purged": [], "disabled": True}
        last = paths.META_DIR / "retention/last.json"
        if not force and last.exists() and time.time() - last.stat().st_mtime < 86400: return {"skipped": "今日已执行"}
        moved = []; purged = []; failures = []
        for row in plan["candidates"]:
            source = paths.RUNS_DIR / row["path"]
            if source.resolve() in protected_runs(): continue  # 执行前重新核对引用
            safe_tree(source, paths.RUNS_DIR)
            identifier = uuid4().hex; folder = paths.META_DIR / "run_trash" / identifier
            folder.mkdir(parents=True); safe_tree(folder, paths.META_DIR / "run_trash")
            record = {"id": identifier, "path": row["path"], "bytes": row["bytes"], "moved_at": time.time()}
            write_json(folder / "record.json", record); source.rename(folder / "payload"); moved.append(record)
        for record in trash():
            if time.time() - record["moved_at"] < settings.trash_days * 86400: continue
            folder = paths.META_DIR / "run_trash" / record["id"]
            safe_tree(folder, paths.META_DIR / "run_trash")
            try: shutil.rmtree(folder); purged.append(record["id"])
            except OSError as error: failures.append(str(error))
        result = {"checked_at": datetime.now(timezone.utc).isoformat(), "moved": moved, "purged": purged, "failures": failures}
        write_json(last, result); return result


def start_scheduler():
    def loop():
        while True:
            try: sweep()
            except Exception:
                import logging
                logging.getLogger(__name__).exception("自动保留策略执行失败")
            time.sleep(3600)
    threading.Thread(target=loop, daemon=True, name="run-retention").start()


def main():
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--apply", action="store_true"); parser.add_argument("--restore")
    args = parser.parse_args()
    print(json.dumps(restore(args.restore) if args.restore else sweep(force=True) if args.apply else preview(), ensure_ascii=False, indent=2))
