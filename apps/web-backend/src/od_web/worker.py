"""Web 工作进程,前端重载/关闭不会中断它。"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time

from od_platform.common import paths
from od_platform.common.json_io import write_json
from od_platform.runtime_config.mixercseg import MixerCSegConfig

from .jobs import utc_now


def run(job_dir: Path) -> int:
    request = json.loads((job_dir / "request.json").read_text(encoding="utf-8"))
    if request["kind"] == "training":
        from od_platform.training.experiment import run_experiment
        return run_experiment(job_dir)
    from od_platform.inference.cancel import CancelToken
    from od_platform.inference.hooks import InferHooks
    from od_platform.inference.mixercseg_service import infer_mixercseg
    state = json.loads((job_dir / "status.json").read_text(encoding="utf-8"))
    state.update(status="running", phase="inference", pid=os.getpid(), started_at=utc_now())
    write_json(job_dir / "status.json", state)
    start = time.perf_counter()
    token = CancelToken()
    def on_frame(event):
        state.update(batch=event.frame_idx + 1, progress=0.5, elapsed_seconds=time.perf_counter() - start)
        write_json(job_dir / "status.json", state)
        if (job_dir / "cancel.request").exists(): token.cancel()
    try:
        if (job_dir / "cancel.request").exists(): token.cancel()
        config = MixerCSegConfig(**request["config"])
        result = infer_mixercseg(config, source=request["source"], show=False, save=True,
                                 show_info=False, hooks=InferHooks(on_frame=on_frame), cancel_token=token)
        if not result.success:
            raise RuntimeError(result.error)
        state.update(status="cancelled" if token.is_cancelled() else "completed", phase="completed", progress=1.0,
                     run_dir=result.output_dir.relative_to(paths.ROOT_DIR).as_posix(),
                     stats=result.stats, elapsed_seconds=time.perf_counter() - start, finished_at=utc_now())
    except Exception as e:
        logging.exception("推理任务失败")
        state.update(status="failed", phase="failed", error=str(e), finished_at=utc_now())
    write_json(job_dir / "status.json", state)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--job-dir", type=Path, required=True)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(run(arguments.job_dir))
