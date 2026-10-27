"""远程 GPU 入口:任务独立运行,SSH 断开不影响计算。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import json
import logging
import os
import subprocess
import sys
import threading
import tarfile

from od_platform.common import paths
from od_platform.common.json_io import write_json


def hardware() -> dict:
    # 先加载 torch 的动态库,再检查原版 CUDA 扫描内核。
    from .app import _local_hardware
    result = _local_hardware()
    if result["cuda_available"] and result["gpu_memory_gb"] is None:
        try:
            output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                              "--format=csv,noheader,nounits", "-i", "0"], timeout=5, text=True)
            used, memory, total = [float(value.strip()) for value in output.strip().split(",")]
            result.update(gpu_percent=used, gpu_memory_used_gb=round(memory / 1024, 1), gpu_memory_gb=round(total / 1024, 1))
        except (OSError, subprocess.SubprocessError, ValueError): pass
    try:
        importlib.import_module("selective_scan_cuda_oflex")
        result["scan_available"] = True
    except (ImportError, OSError):
        result["scan_available"] = False
    result.update(execution="remote", connected=True)
    return result


@contextmanager
def job_lock():
    import fcntl
    directory = paths.META_DIR / "web"; directory.mkdir(parents=True, exist_ok=True)
    with (directory / "gpu.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try: yield
        finally: fcntl.flock(lock, fcntl.LOCK_UN)


def launch(job_id: str) -> dict:
    import psutil
    from .jobs import JobStore, utc_now
    store = JobStore(); directory = store.directory(job_id)
    with job_lock():
        state = store.read(job_id)
        marker = directory / "launch.json"
        if marker.exists():
            # 重连只接回原任务;即使它已经结束也不重跑。
            previous: dict = json.loads(marker.read_text(encoding="utf-8")); return previous
        for other in store.list():
            if other["id"] != job_id and other["status"] in ("queued", "running", "cancelling"):
                if other.get("pid") and psutil.pid_exists(other["pid"]):
                    raise RuntimeError("远程 GPU 已有运行任务")
        if (directory / "cancel.request").exists():
            state.update(status="cancelled", phase="cancelled", finished_at=utc_now())
            write_json(directory / "status.json", state)
            result: dict = {"cancelled": True}; write_json(marker, result)
            return result
        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        device = hardware()
        if not device["cuda_available"] or (request["config"].get("backend") != "yolo" and not device["scan_available"]):
            raise RuntimeError("远程 CUDA 或所需扫描内核未就绪")
        if request["config"]["device"] != "cuda" or request["config"]["scan_backend"] != "reference":
            raise ValueError("远程任务必须使用 CUDA 和原版扫描内核")
        if request["kind"] == "training":
            from od_platform.common import refs
            from od_platform.data_validation.semantic import validate_semantic_dataset
            if request["config"].get("backend") == "yolo":
                from od_platform.data_pipeline.wsodd import validate
                detection_report = validate(refs.resolve_dataset_yaml(request["config"]["data"]))
                severity, fingerprint = detection_report["overall_severity"], detection_report["data_fingerprint"]
            else:
                report = validate_semantic_dataset(refs.resolve_dataset_yaml(request["config"]["data"]))
                severity, fingerprint = report.overall_severity, report.snapshot.data_fingerprint
            if severity == "ERROR": raise ValueError("远程数据集身份或质检不通过")
            if fingerprint != request["dataset_fingerprint"]:
                raise ValueError("远程与本机数据集指纹不一致")
        state["config"] = request["config"]
        write_json(directory / "status.json", state)
        with (directory / "worker.log").open("ab") as log:
            process = subprocess.Popen([sys.executable, "-m", "od_web.remote_agent", "run", job_id],
                                       cwd=paths.ROOT_DIR, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=log, start_new_session=True)
        result = {"pid": process.pid, "started_at": utc_now()}
        write_json(directory / "process.json", {"pid": process.pid})
        write_json(marker, result)
        return result


def run(job_id: str) -> int:
    from .jobs import JobStore
    from .worker import run as execute
    directory = JobStore().directory(job_id)
    stopped = threading.Event()
    def sample():
        while not stopped.is_set():
            try: write_json(directory / "hardware.json", hardware())
            except Exception: pass
            stopped.wait(4)
    sampler = threading.Thread(target=sample, daemon=True); sampler.start()
    try: return execute(directory)
    except (Exception, KeyboardInterrupt) as error:
        from .jobs import utc_now
        logging.exception("远程工作进程失败")
        state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        state.update(status="failed", phase="failed", error=str(error), finished_at=utc_now())
        write_json(directory / "status.json", state)
        return 1
    finally:
        stopped.set(); sampler.join(timeout=5)


def snapshot(job_id: str) -> dict:
    from .jobs import JobStore
    return JobStore().read(job_id)


def dataset_ready(job_id):
    """相同数据身份且现存文件通过核验才能跳过重新上传;启动时仍做全量核验。"""
    import re
    from .jobs import JobStore
    from od_platform.data_pipeline.wsodd import validate
    request = json.loads((JobStore().directory(job_id) / "request.json").read_text(encoding="utf-8"))
    identifier = request["config"]["data"]
    if request["config"].get("backend") != "yolo" or not re.fullmatch(r"wsodd|ds_[0-9a-f]{16}", identifier):
        raise ValueError("非法目标检测数据集 ID")
    try:
        report = validate(paths.dataset_yaml_path(identifier), use_cache=True, write_report=False)
        return {"ready": report["overall_severity"] != "ERROR" and report["data_fingerprint"] == request.get("dataset_fingerprint")}
    except (OSError, ValueError, KeyError):
        return {"ready": False}


def unpack(job_id):
    from .jobs import JobStore
    from od_platform.data_pipeline.import_dataset import safe_relative
    import shutil
    import re
    from od_platform.data_pipeline.detection_import import MAX_EXPANDED, MAX_FILES
    directory = JobStore().directory(job_id)
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    identifier = request["config"]["data"]
    from od_platform.runtime_config.detection import DetectionConfig
    architecture = DetectionConfig(**request["config"]).architecture
    if request["config"].get("backend") != "yolo" or not re.fullmatch(r"wsodd|ds_[0-9a-f]{16}", identifier):
        raise ValueError("非法目标检测数据集 ID")
    size = 0; seen = set()
    with tarfile.open(directory / "dataset.tar") as archive:
        for member in archive:
            relative = safe_relative(member.name)
            if not member.isfile() or member.size > 64*1024**2: raise ValueError("非法检测数据包成员")
            size += member.size
            if relative in seen or len(seen) >= MAX_FILES + 3 or size > MAX_EXPANDED: raise ValueError("检测数据包重复或超过限制")
            seen.add(relative)
            if not (relative.as_posix().startswith(f"data/processed/{identifier}/") or relative.as_posix() in (f"apps/platform/configs/datasets/{identifier}.yaml", f"models/pretrained/{architecture}")):
                raise ValueError("检测数据包路径不被允许")
            target = paths.ROOT_DIR / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("wb") as output: shutil.copyfileobj(source, output)
    return {"unpacked": True}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=["hardware", "check", "launch", "run", "snapshot", "unpack", "dataset_ready"])
    parser.add_argument("job_id", nargs="?"); args = parser.parse_args()
    if args.action == "hardware": result = hardware()
    elif args.action == "check":
        result = hardware()
        importlib.import_module("ultralytics")
        importlib.import_module("od_platform.training.detection_experiment")
        if not result["cuda_available"]: raise RuntimeError("远程 CUDA 未就绪")
        result["runtime_ready"] = True
    elif args.action == "launch": result = launch(args.job_id)
    elif args.action == "snapshot": result = snapshot(args.job_id)
    elif args.action == "unpack": result = unpack(args.job_id)
    elif args.action == "dataset_ready": result = dataset_ready(args.job_id)
    else: raise SystemExit(run(args.job_id))
    print("ODP_JSON=" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
