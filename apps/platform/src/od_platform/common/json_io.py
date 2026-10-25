"""运行状态原子写入:监控端不会读到半份 JSON。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from uuid import uuid4


def write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.01 * (attempt + 1))  # Windows 读者短暂持有文件时等待释放
    finally:
        temporary.unlink(missing_ok=True)
