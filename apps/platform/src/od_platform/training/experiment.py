"""独立工作进程:完整训练结束后自动执行 test 验收,状态可被 Web 端持续读取。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time

from od_platform.common import paths, refs
from od_platform.common.json_io import write_json
from od_platform.common.run_context import RunContext
from od_platform.runtime_config.mixercseg import MixerCSegConfig


def run_experiment(job_dir: Path) -> int:
    from od_platform.training.acceptance import evaluate_checkpoint
    from od_platform.training.mixercseg_service import train_mixercseg
    request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    config = MixerCSegConfig(**request["config"])
    state = {"id": job_dir.name, "kind": "training", "name": request.get("name", "CrackMap 完整实验"),
             "status": "running", "phase": "starting", "pid": os.getpid(),
             "created_at": request["created_at"], "started_at": datetime.now(timezone.utc).isoformat(),
             "config": config.model_dump(), "history": [], "progress": 0.0}
    started = time.perf_counter()
    def cancelled(): return (job_dir / "cancel.request").exists()
    def update(event):
        state.update(event)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        state["elapsed_seconds"] = time.perf_counter() - started
        epoch = state.get("epoch", 0)
        if event.get("phase") == "training":
            state["progress"] = min(0.94, ((epoch - 1) + event["batch"] / event["batches"]) / config.epochs * 0.94)
        elif event.get("phase") == "epoch_complete":
            state["progress"] = epoch / config.epochs * 0.94
        elif event.get("phase") == "testing":
            state["progress"] = 0.94 + event["batch"] / event["batches"] * 0.06
        if state["progress"] > 0:
            state["eta_seconds"] = state["elapsed_seconds"] / state["progress"] * (1 - state["progress"])
        write_json(job_dir / "status.json", state)
    try:
        update({})
        data_yaml = refs.resolve_dataset_yaml(config.data)
        with RunContext("training") as run:
            update({"run_dir": run.run_dir.relative_to(paths.ROOT_DIR).as_posix(), "run_id": run.run_id})
            result = train_mixercseg(config, data_yaml, run, on_progress=update, should_cancel=cancelled)
            if not result.success or result.best_weight is None:
                raise InterruptedError(result.message) if cancelled() else RuntimeError(result.message)
            audit = json.loads((run.run_dir / "odp_audit.json").read_text(encoding="utf-8"))
            if audit["smoke"] or len(audit["history"]) != config.epochs:
                raise ValueError("完整训练验收不允许 smoke 或缺失 epoch")
            update({"best_weight": str(result.best_weight), "phase": "testing"})
            report = evaluate_checkpoint(config, data_yaml, result.best_weight, job_dir / "evaluation",
                                         history=audit["history"], min_f1=request.get("min_f1"),
                                         on_progress=update, should_cancel=cancelled)
            update({"status": "completed", "phase": "completed", "progress": 1.0,
                    "metrics": report["metrics"], "quality_status": report["quality_status"],
                    "finished_at": datetime.now(timezone.utc).isoformat(), "eta_seconds": 0})
        return 0
    except (Exception, KeyboardInterrupt) as e:
        logging.exception("实验失败或被停止")
        update({"status": "cancelled" if cancelled() else "failed", "phase": "cancelled" if cancelled() else "failed",
                "error": str(e), "finished_at": datetime.now(timezone.utc).isoformat()})
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--job-dir", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    sys.exit(run_experiment(arguments.job_dir.resolve()))
