"""任务登记:远程 GPU 执行、本机持久状态、取消与受控产物访问。"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import threading
from uuid import uuid4

import psutil

from od_platform.common import paths
from od_platform.common.json_io import write_json

_LOCK = threading.Lock()
_ID = re.compile(r"^[0-9a-f]{32}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, root: Path | None = None):
        self.root = root or paths.RUNS_DIR / "web_jobs"
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, job_id: str) -> Path:
        if not _ID.fullmatch(job_id):
            raise ValueError("非法任务 ID")
        directory = self.root / job_id
        if not directory.is_dir():
            raise FileNotFoundError("任务不存在")
        return directory

    def read(self, job_id: str) -> dict:
        directory = self.directory(job_id)
        state: dict = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        process_path = directory / "process.json"
        if not state.get("pid") and process_path.exists():
            state["pid"] = json.loads(process_path.read_text(encoding="utf-8"))["pid"]
        if state.get("execution") != "remote" and state["status"] in ("queued", "running", "cancelling"):
            pid = state.get("pid")
            if pid and not psutil.pid_exists(pid):
                state.update(status="failed", phase="failed", error="工作进程已退出,未写入完成结果", finished_at=utc_now())
                write_json(directory / "status.json", state)
        state["cancel_requested"] = (directory / "cancel.request").exists()
        if state.get("phase") == "syncing" and state.get("kind") == "training" and state.get("config", {}).get("backend") == "yolo":
            # 回传进程可能长时间下载大图;按本机已落盘文件计算进度,不再表现成 GPU 仍在训练。
            root = (paths.ROOT_DIR / state.get("run_dir", "runs/nonexistent")).resolve()
            if root.is_relative_to(paths.RUNS_DIR.resolve()):
                evaluation = directory / "evaluation"
                files = [file for folder in (root, evaluation) for file in folder.rglob("*") if file.is_file()]
                total = (state.get("dataset_counts") or {}).get("test", 48)
                received = 0
                for file in files:
                    try: received += file.stat().st_size
                    except FileNotFoundError: pass  # .download 可能刚刚原子改名
                state["sync_progress"] = {"comparisons": len(list((evaluation / "comparisons").glob("*.png"))),
                                          "expected_comparisons": min(48, total), "received_bytes": received}
        return state

    def list(self) -> list[dict]:
        rows = []
        for directory in self.root.iterdir():
            if directory.is_dir() and _ID.fullmatch(directory.name) and (directory / "status.json").exists():
                state = self.read(directory.name)
                from .remote import ensure_bridge
                ensure_bridge(directory, state)
                rows.append(state)
        return sorted(rows, key=lambda row: row["created_at"], reverse=True)

    def create(self, request: dict) -> dict:
        with _LOCK:
            if any(row["status"] in ("queued", "running", "cancelling") for row in self.list()):
                raise RuntimeError("当前已有运行任务,请完成或停止后再提交")
            job_id = uuid4().hex
            directory = self.root / job_id; directory.mkdir()
            request = {**request, "config": {**request["config"], "device": "cuda", "scan_backend": "reference"},
                       "require_cuda": True, "created_at": utc_now()}
            write_json(directory / "request.json", request)
            state = {"id": job_id, "kind": request["kind"], "status": "queued", "phase": "queued",
                     "created_at": request["created_at"], "config": request["config"], "progress": 0.0,
                     "execution": "remote", "remote_managed": True, "connection_status": "connecting",
                     "history": [], "name": request.get("name", "CrackMap 完整实验")}
            state["media_type"] = request.get("media_type", "image")
            write_json(directory / "status.json", state)
            from .remote import start_bridge
            try: start_bridge(directory)
            except OSError as e:
                state.update(status="failed", phase="failed", error=str(e)); write_json(directory / "status.json", state)
                raise
            return self.read(job_id)

    def cancel(self, job_id: str) -> dict:
        state = self.read(job_id)
        if state["status"] not in ("queued", "running", "cancelling"):
            raise ValueError("任务已结束")
        (self.directory(job_id) / "cancel.request").touch()
        return {**state, "cancel_requested": True}

    def artifact(self, job_id: str, relative: str) -> Path:
        directory = self.directory(job_id)
        state = self.read(job_id)
        root = directory
        if relative.startswith("run/"):
            if not state.get("run_dir"):
                raise FileNotFoundError("运行产物尚未生成")
            root = (paths.ROOT_DIR / state["run_dir"]).resolve()
            if not root.is_relative_to(paths.RUNS_DIR.resolve()):
                raise ValueError("运行目录超出 runs")
            relative = relative[4:]
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
            raise FileNotFoundError("产物不存在")
        if candidate.suffix.lower() not in (".json", ".jsonl", ".mp4", ".csv", ".md", ".png", ".jpg", ".pth", ".pt", ".log"):
            raise ValueError("不支持的产物格式")
        return candidate
