"""视频摘要在本机调用 DeepSeek;密钥和原视频均不发送到对方模型。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from uuid import uuid4

import psutil

from od_platform.common import paths
from od_platform.common.json_io import write_json
from . import assistant
from .jobs import JobStore, utc_now

_LOCK = threading.Lock()


def read(store, identifier):
    file = store.directory(identifier) / "explanation.json"
    value = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {"status": "pending"}
    process_file = file.with_name("explanation_process.json")
    if value["status"] == "running" and not value.get("pid") and process_file.exists():
        process = json.loads(process_file.read_text())
        if process.get("generation") == value.get("generation"): value["pid"] = process["pid"]
    if value["status"] == "running" and value.get("pid") and not psutil.pid_exists(value["pid"]):
        value.update(status="failed", error="解释进程已退出,可重试")
    return value


def start(store, identifier):
    with _LOCK:
        state = store.read(identifier)
        if state["kind"] != "inference" or state["status"] != "completed" or state.get("media_type") != "video":
            raise ValueError("视频目标检测完成后才能生成解释")
        previous = read(store, identifier)
        if previous["status"] in ("running", "completed"): return previous
        if not assistant.status()["configured"]: raise ValueError("智能助手尚未启用,请由管理员完成服务配置")
        directory = store.directory(identifier)
        value = {"status": "running", "generation": uuid4().hex, "started_at": utc_now()}
        write_json(directory / "explanation.json", value)
        try:
            with (directory / "explanation.log").open("ab") as log:
                process = subprocess.Popen([sys.executable, "-m", "od_web.explanation", "--job-dir", str(directory)], cwd=paths.ROOT_DIR,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            write_json(directory / "explanation_process.json", {"pid": process.pid, "generation": value["generation"]})
        except OSError:
            write_json(directory / "explanation.json", {"status": "failed", "error": "解释进程无法启动,请重试"}); raise
        return read(store, identifier)


def run(directory):
    import time
    store = JobStore(directory.parent); target = directory / "explanation.json"
    for _ in range(100):
        if target.exists(): break
        time.sleep(0.05)
    running = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
    write_json(target, {**running, "status": "running", "pid": os.getpid()})
    try:
        summary_file = store.artifact(directory.name, "run/video_summary.json")
        if summary_file.stat().st_size > 128*1024: raise ValueError("视频摘要过大")
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        evidence = {"project": "港区水域通航目标检测", "video_analysis": {key: summary.get(key) for key in
                     ("media_type", "frames", "fps", "duration_seconds", "class_summary", "threshold", "max_detections_per_frame", "note")}}
        result = assistant.ask("请根据视频检测摘要解释出现的目标类别、出现频率、置信度及局限；不要把逐帧计数当作独立目标数量，不推断航速、轨迹或碰撞风险。", evidence)
        value = {"status": "completed", "finished_at": utc_now(), **result}
    except Exception as error:
        value = {"status": "failed", "finished_at": utc_now(), "error": str(error) if isinstance(error, (ValueError, assistant.ProviderError)) else "解释失败,请重试"}
    write_json(target, value); return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--job-dir", type=Path, required=True)
    run(parser.parse_args().job_dir)
