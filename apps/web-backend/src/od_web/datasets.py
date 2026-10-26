"""Django 数据集登记与工作进程启动,只向引擎传递受控 ID。"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

import psutil

from od_platform.common import paths
from od_platform.common.semantic import load_semantic_dataset
from od_platform.data_pipeline.import_dataset import MAX_FILES, safe_relative
from od_platform.data_validation.semantic import validate_semantic_dataset

from .models import Dataset

MAX_UPLOAD = 512 * 1024**2


def dataset_yaml(identifier: str) -> Path:
    if identifier != "crackmap" and not Dataset.objects.filter(pk=identifier).exists():
        raise ValueError("数据集不存在,请先导入数据集")
    return paths.dataset_yaml_path(identifier)


def describe(identifier: str) -> dict:
    record = Dataset.objects.filter(pk=identifier).first()
    if record is None and identifier != "crackmap": raise ValueError("数据集不存在")
    if record and record.status == "importing" and record.pid and not psutil.pid_exists(record.pid):
        record.status = "failed"; record.error = "导入进程已退出,请查看格式说明并重新导入"; record.save(update_fields=["status", "error"])
    value: dict = {"id": identifier, "name": record.name if record else "CrackMap", "status": record.status if record else "ready",
             "phase": record.phase if record else "已有数据集", "progress": record.progress if record else 1,
             "source_name": record.source_name if record else "MixerCSeg / CrackMap", "error": record.error if record else "",
             "available": False, "splits": {"train": 0, "val": 0, "test": 0}, "quality": record.report if record else None}
    yaml_path = paths.dataset_yaml_path(identifier)
    if yaml_path.exists():
        document, root = load_semantic_dataset(yaml_path)
        value.update(available=root.exists(), fingerprint=document["data_fingerprint"],
                     splits={s: len(list((root / "images" / s).glob("*.png"))) for s in ("train", "val", "test")})
        quality_path = paths.RUNS_DIR / "data_validation" / identifier / "report.json"
        if quality_path.exists(): value["quality"] = json.loads(quality_path.read_text(encoding="utf-8"))
        if value["quality"] and value["quality"]["overall_severity"] == "ERROR": value["status"] = "blocked"
    value["trainable"] = value["available"] and value["status"] == "ready" and bool(value["quality"])
    return value


def listing() -> list[dict]:
    return [describe("crackmap"), *[describe(key) for key in Dataset.objects.values_list("id", flat=True)]]


def validate(identifier: str) -> dict:
    report = validate_semantic_dataset(dataset_yaml(identifier), run_id=identifier).to_dict()
    Dataset.objects.filter(pk=identifier).update(report=report, status="blocked" if report["overall_severity"] == "ERROR" else "ready")
    return report


def launch_import(record: Dataset, source: Path, mode: str) -> None:
    with (source.parent / "import.log").open("ab") as log:
        process = subprocess.Popen([sys.executable, "-m", "od_web.dataset_worker", "--dataset", record.id,
                                   "--source", str(source), "--mode", mode], cwd=paths.ROOT_DIR,
                                   stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    Dataset.objects.filter(pk=record.pk).update(pid=process.pid)


def receive(request, name: str, mode: str) -> dict:
    files = request.FILES.getlist("files")
    if not files or len(files) > MAX_FILES: raise ValueError("请选择数据集文件,最多 20000 个文件")
    if mode == "archive" and (len(files) != 1 or not files[0].name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))):
        raise ValueError("请选择一个 ZIP、TAR、TAR.GZ 或 TGZ 文件")
    relative = json.loads(request.POST.get("relative_paths", "[]")) if mode == "folder" else ["dataset.archive"]
    if not isinstance(relative, list) or len(relative) != len(files) or not all(isinstance(value, str) for value in relative): raise ValueError("文件夹相对路径与文件数量不一致")
    entries = [safe_relative(value) for value in relative]
    if len({str(value).casefold() for value in entries}) != len(entries): raise ValueError("上传文件路径重复")
    if sum(file.size for file in files) > MAX_UPLOAD: raise ValueError("单次上传总大小不能超过 512 MB")
    identifier = "ds_" + uuid4().hex[:16]
    directory = paths.META_DIR / "dataset_uploads" / identifier; incoming = directory / "incoming"
    incoming.mkdir(parents=True, exist_ok=False)
    record = Dataset.objects.create(id=identifier, name=name, source_name=files[0].name if mode == "archive" else "文件夹上传")
    try:
        total = 0
        for file, relative_path in zip(files, entries, strict=True):
            destination = incoming / relative_path; destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                for chunk in file.chunks():
                    total += len(chunk)
                    if total > MAX_UPLOAD: raise ValueError("上传超过 512 MB")
                    stream.write(chunk)
        launch_import(record, incoming / entries[0] if mode == "archive" else incoming, mode)
    except Exception as error:
        record.status = "failed"; record.error = str(error); record.save(update_fields=["status", "error"])
        raise
    return describe(identifier)
