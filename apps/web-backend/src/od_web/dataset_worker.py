"""独立导入进程:解压/转换/质检不会阻塞页面请求。"""
import argparse
import os
from pathlib import Path


def run(identifier: str, source: Path, mode: str) -> None:
    from od_platform.data_pipeline.import_dataset import extract_archive, import_semantic_dataset
    from od_platform.data_validation.semantic import validate_semantic_dataset
    from od_platform.common import paths
    from .models import Dataset
    record = Dataset.objects.get(pk=identifier)
    try:
        if mode == "archive":
            Dataset.objects.filter(pk=identifier).update(phase="检查并解压", progress=0.1)
            destination = source.parent.parent / "extracted"
            extract_archive(source, destination); source = destination
        Dataset.objects.filter(pk=identifier).update(phase="转换图像与标注", progress=0.2)
        def progress(done, total):
            if done == total or done % 10 == 0:
                Dataset.objects.filter(pk=identifier).update(progress=0.2 + 0.6 * done / total)
        yaml_path = import_semantic_dataset(source, identifier, on_progress=progress)
        Dataset.objects.filter(pk=identifier).update(phase="数据质量检查", progress=0.85)
        report = validate_semantic_dataset(yaml_path, run_id=identifier).to_dict()
        import yaml
        document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        counts = {s: len(list((paths.PROCESSED_DATA_DIR / identifier / "images" / s).glob("*.png"))) for s in ("train", "val", "test")}
        Dataset.objects.filter(pk=identifier).update(status="blocked" if report["overall_severity"] == "ERROR" else "ready",
                     phase="检查完成", progress=1, report=report, counts=counts, fingerprint=document["data_fingerprint"], error="")
    except Exception as error:
        Dataset.objects.filter(pk=record.pk).update(status="failed", phase="导入失败", error=str(error))
        raise


if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "od_web.settings")
    import django
    django.setup()
    parser = argparse.ArgumentParser(); parser.add_argument("--dataset", required=True)
    parser.add_argument("--source", type=Path, required=True); parser.add_argument("--mode", required=True)
    args = parser.parse_args(); run(args.dataset, args.source, args.mode)
