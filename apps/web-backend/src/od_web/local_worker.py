"""本机 CPU 推理工作进程，关闭网页后仍继续运行。"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .jobs import JobStore
from .worker import run


def main(directory: Path) -> int:
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if request.get("execution") != "local" or request.get("kind") != "inference" or request["config"].get("backend") != "yolo" or request["config"].get("device") != "cpu":
        raise ValueError("本机工作进程仅接受显式指定的 CPU 目标检测")
    result = run(directory)
    if result == 0 and request.get("media_type") == "video":
        from . import explanation
        try: explanation.start(JobStore(directory.parent), directory.name)
        except (ValueError, RuntimeError): pass  # 推理完成独立于可重试的解释。
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main(args.job_dir.resolve()))
