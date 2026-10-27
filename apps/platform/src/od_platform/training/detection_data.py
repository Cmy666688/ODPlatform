"""隔离训练引擎可能改写的 JPEG/标签缓存,保留冻结的数据契约。"""
import shutil

import yaml

from od_platform.common.json_io import write_json
from od_platform.common.semantic import file_sha256


def stage_dataset(document, root, directory, update):
    target = directory / "engine_dataset"
    if target.exists(): raise ValueError("训练数据副本已存在,不能覆盖运行现场")
    repairs = []; images = sorted((root / "images").rglob("*.jpg"))
    for index, source in enumerate(images):
        output = target / source.relative_to(root); output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
        # PIL 已能完整解码的 JPEG 有时尾部缺少 EOI;只补结束标记,不重新压缩像素。
        with output.open("rb+") as stream:
            stream.seek(-2, 2)
            if stream.read(2) != b"\xff\xd9":
                stream.seek(0, 2); stream.write(b"\xff\xd9")
                repairs.append({"file": source.relative_to(root).as_posix(), "source_sha256": file_sha256(source),
                                "engine_sha256": file_sha256(output), "operation": "append JPEG EOI; decoded pixels unchanged"})
        if index % 100 == 0 or index+1 == len(images): update({"phase": "staging", "staged_images": index+1, "staging_total": len(images)})
    shutil.copytree(root / "labels", target / "labels", ignore=shutil.ignore_patterns("*.cache"))
    runtime = {key: document[key] for key in ("train", "val", "test", "names")}; runtime["path"] = str(target)
    data_yaml = directory / "engine_dataset.yaml"
    data_yaml.write_text(yaml.safe_dump(runtime, sort_keys=False, allow_unicode=True), encoding="utf-8")
    audit = {"data_fingerprint": document["data_fingerprint"], "source_unchanged": True, "jpeg_eoi_appended": repairs,
             "policy": "independent file copies; original split, labels and class mapping retained"}
    write_json(directory / "engine_dataset_audit.json", audit)
    return data_yaml, audit
