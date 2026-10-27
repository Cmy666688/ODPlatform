"""YOLO 检测实验:逐轮监控、独立 test、类别指标、GT/预测可视化与身份审计。"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import yaml

from od_platform.common import paths
from od_platform.common.json_io import write_json
from od_platform.common.run_context import RunContext
from od_platform.common.semantic import file_sha256, read_bgr, write_png
from od_platform.data_pipeline.wsodd import validate
from od_platform.runtime_config.detection import DetectionConfig


def now(): return datetime.now(timezone.utc).isoformat()


def loss_total(value):
    if isinstance(value, dict): return sum(loss_total(item) for item in value.values())
    return float(value.sum().item()) if hasattr(value, "sum") else float(value)


def epoch_record(trainer, epochs, recorded):
    epoch = trainer.epoch + 1
    # 上游在最终 best 验证时会再次触发回调,且 epoch 可能已经自增。
    if epoch <= recorded or epoch > epochs: return None
    values = trainer.metrics
    p, r = float(values.get("metrics/precision(B)", 0)), float(values.get("metrics/recall(B)", 0))
    return {"epoch": epoch, "train_loss": loss_total(trainer.tloss), "train_samples": len(trainer.train_loader.dataset),
            "validation": {"loss": sum(float(v) for k, v in values.items() if k.startswith("val/")),
            "precision": p, "recall": r, "F1": 2*p*r/(p+r) if p+r else 0, "mAP50": float(values.get("metrics/mAP50(B)", 0)),
            "mAP50_95": float(values.get("metrics/mAP50-95(B)", 0))}}


def metrics_summary(result):
    precision, recall, ap50, ap = [float(value) for value in result.box.mean_results()]
    return {"precision": precision, "recall": recall, "mAP50": ap50, "mAP50_95": ap,
            "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def dataset_file(reference, directory):
    original = paths.dataset_yaml_path(reference)
    document = yaml.safe_load(original.read_text(encoding="utf-8"))
    root = (paths.ROOT_DIR / document["path"]).resolve()
    if not root.is_relative_to(paths.PROCESSED_DATA_DIR.resolve()): raise ValueError("检测数据目录超出 processed")
    resolved = {key: document[key] for key in ("train", "val", "test", "names")}
    resolved["path"] = str(root)
    target = directory / "dataset.yaml"; target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return original, target, document, root


def draw_boxes(image, rows, names):
    import cv2
    output = image.copy(); height, width = output.shape[:2]
    for category, x, y, w, h in rows:
        left, top, right, bottom = int((x-w/2)*width), int((y-h/2)*height), int((x+w/2)*width), int((y+h/2)*height)
        cv2.rectangle(output, (left, top), (right, bottom), (44, 170, 100), 2)
        cv2.putText(output, str(names[int(category)]), (left, max(16, top-4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (44, 170, 100), 1)
    return output


def curves(history, target):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["validation"]["loss"] for row in history], label="Validation")
    for key in ("mAP50", "mAP50_95"):
        axes[1].plot(epochs, [row["validation"][key] for row in history], label=key)
    for axis in axes: axis.set_xlabel("Epoch"); axis.legend(); axis.grid(alpha=0.2)
    figure.tight_layout(); figure.savefig(target, dpi=140); plt.close(figure)


def evaluate(config, best, data_yaml, document, root, directory, history, update, cancelled):
    from ultralytics import YOLO
    model = YOLO(str(best))
    def check(_):
        if cancelled(): raise InterruptedError("已请求停止测试")
    model.add_callback("on_val_batch_end", check)
    result = model.val(data=str(data_yaml), split="test", device=0, imgsz=config.imgsz,
                       batch=config.batch_size_test, workers=config.num_threads, conf=0.001, iou=config.iou,
                       project=str(directory.parent), name=directory.name, exist_ok=True, plots=True, save_json=True, verbose=False)
    metric = metrics_summary(result)
    manifest = json.loads((paths.ROOT_DIR / document["manifest_path"]).read_text(encoding="utf-8"))
    samples = [row for row in manifest["samples"] if row["split"] == "test"]
    supported = {int(category): index for index, category in enumerate(result.box.ap_class_index)}
    classes = []
    for category, name in enumerate(document["names"].values()):
        index = supported.get(category); instances = sum(row["classes"].count(category) for row in samples)
        classes.append({"class_id": category, "name": name, "instances": instances,
                        "precision": float(result.box.p[index]) if index is not None else None,
                        "recall": float(result.box.r[index]) if index is not None else None,
                        "AP50": float(result.box.ap50[index]) if index is not None else None,
                        "AP50_95": float(result.box.ap[index]) if index is not None else None})
    report = {"task": "detect", "dataset": config.data, "images": len(samples), "metrics": metric, "classes": classes,
              "checkpoint_sha256": file_sha256(best), "data_fingerprint": document["data_fingerprint"],
              "quality_status": "needs_review", "metric_policy": "Ultralytics detection AP at confidence sweep; IoU 0.50:0.95; macro precision/recall; F1 is harmonic mean of macro P/R, not fixed-threshold segmentation F1",
              "visualization_confidence": config.threshold, "validation_confidence_floor": 0.001,
              "split": "test", "note": "test 不参与选模;缺失类别的 AP 为空,不以总体均值填充"}
    write_json(directory / "report.json", report)
    with (directory / "per_class.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(classes[0])); writer.writeheader(); writer.writerows(classes)
    curves(history, directory / "training_curves.png")
    # 全量 test 计算指标,最多 48 张分层展示,避免同步数千张图片阻塞完成状态。
    chosen = sorted(samples, key=lambda row: (len(row["classes"]), row["stem"]))
    chosen = chosen[::max(1, len(chosen)//48)][:48]
    visualized = []
    for index, sample in enumerate(chosen):
        if cancelled(): raise InterruptedError("已请求停止可视化")
        image = root / "images/test" / (sample["stem"] + ".jpg")
        labels = root / "labels/test" / (sample["stem"] + ".txt")
        truth = [[float(v) for v in line.split()] for line in labels.read_text().splitlines()]
        prediction = model.predict(source=str(image), device=0, imgsz=config.imgsz, conf=config.threshold,
                                   iou=config.iou, save=False, verbose=False)[0]
        comparison = np.hstack([draw_boxes(read_bgr(image), truth, document["names"]), prediction.plot()])
        write_png(directory / "comparisons" / (sample["stem"] + ".png"), comparison)
        visualized.append(sample["stem"])
        update({"phase": "visualizing", "batch": index + 1, "batches": len(chosen), "progress": 0.97 + 0.03*(index+1)/len(chosen)})
    report["visualized_images"] = visualized; write_json(directory / "report.json", report)
    (directory / "report.md").write_text(f"# {config.data} 检测验收\n\n" + "\n".join(f"- {key}: {value:.6f}" for key, value in metric.items())
                                           + "\n\n完整 test 指标;图像左侧为 GT，右侧为预测。未设业务目标，效果待人工评审。\n", encoding="utf-8")
    return report


def run(job_dir: Path) -> int:
    import torch
    from ultralytics import YOLO
    from od_platform.runtime_config.train import YOLOTrainConfig
    request = json.loads((job_dir / "request.json").read_text(encoding="utf-8")); config = DetectionConfig(**request["config"])
    state = {"id": job_dir.name, "kind": request["kind"], "name": request["name"], "status": "running", "phase": "starting",
             "pid": __import__("os").getpid(), "created_at": request["created_at"], "started_at": now(), "config": config.model_dump(),
             "history": [], "progress": 0, "dataset_name": request.get("dataset_name", "WSODD"),
             "dataset_counts": request.get("dataset_counts"), "dataset_fingerprint": request.get("dataset_fingerprint"), "task": "detect"}
    start = time.perf_counter()
    def cancelled(): return (job_dir / "cancel.request").exists()
    def update(event):
        state.update(event); state.update(updated_at=now(), elapsed_seconds=time.perf_counter()-start)
        if 0 < state["progress"] < 1: state["eta_seconds"] = state["elapsed_seconds"]*(1-state["progress"])/state["progress"]
        write_json(job_dir / "status.json", state)
    try:
        if config.device != "cuda" or not torch.cuda.is_available(): raise RuntimeError("检测实验必须在远程 CUDA GPU 执行")
        torch.set_num_threads(config.torch_threads); update({})
        if request["kind"] == "inference": return infer(config, request["source"], job_dir, state, update, cancelled)
        report = validate(paths.dataset_yaml_path(config.data))
        if report["overall_severity"] == "ERROR" or report["data_fingerprint"] != request["dataset_fingerprint"]: raise ValueError("检测数据质检或身份不通过")
        with RunContext("training") as context:
            original, data_yaml, document, root = dataset_file(config.data, job_dir)
            from .detection_data import stage_dataset
            data_yaml, data_audit = stage_dataset(document, root, job_dir, update)
            update({"run_dir": context.run_dir.relative_to(paths.ROOT_DIR).as_posix(), "phase": "training"})
            pretrained = paths.PRETRAINED_MODELS_DIR / config.architecture
            model = YOLO(str(pretrained) if pretrained.is_file() else config.architecture)
            batch_count = 0
            def batch_end(trainer):
                nonlocal batch_count
                batch_count += 1
                if cancelled(): raise InterruptedError("已请求停止训练")
                if batch_count % 10 == 0:
                    batches = len(trainer.train_loader)
                    update({"epoch": trainer.epoch+1, "batch": (batch_count-1)%batches+1, "batches": batches,
                            "progress": 0.94*(trainer.epoch+((batch_count-1)%batches+1)/batches)/config.epochs})
            def epoch_end(trainer):
                record = epoch_record(trainer, config.epochs, len(state["history"]))
                if record is None: return
                state["history"].append(record)
                update({"epoch": trainer.epoch+1, "phase": "training", "progress": 0.94*(trainer.epoch+1)/config.epochs})
            model.add_callback("on_train_batch_end", batch_end); model.add_callback("on_fit_epoch_end", epoch_end)
            recipe = YOLOTrainConfig(task="detect", model=config.architecture, data=str(data_yaml), device="0", epochs=config.epochs,
                                     batch=config.batch_size_train, imgsz=config.imgsz, lr0=config.lr, optimizer=config.optimizer,
                                     weight_decay=config.weight_decay, seed=config.seed, workers=config.num_threads, patience=0,
                                     cos_lr=config.cos_lr, warmup_bias_lr=config.warmup_bias_lr, mosaic=config.mosaic,
                                     scale=config.scale, hsv_s=config.hsv_s, close_mosaic=min(config.close_mosaic, config.epochs),
                                     cache=config.cache, rect=config.rect, deterministic=config.deterministic,
                                     amp=True, plots=True, save=True, pretrained=True)
            kwargs = recipe.to_ultralytics_kwargs(); kwargs.pop("model", None)
            kwargs.update(project=str(context.run_dir.parent), name=context.run_id, exist_ok=True)
            model.train(**kwargs)
            if cancelled(): raise InterruptedError("已停止训练")
            if len(state["history"]) != config.epochs: raise ValueError("完整训练轮次不齐")
            best = context.run_dir / "weights/best.pt"
            update({"phase": "testing", "best_weight": str(best), "progress": 0.94})
            evaluation = evaluate(config, best, data_yaml, document, root, job_dir / "evaluation", state["history"], update, cancelled)
            write_json(context.run_dir / "odp_audit.json", {"task": "detect", "config": config.model_dump(), "resolved_device": "cuda",
                       "history": state["history"], "data_fingerprint": document["data_fingerprint"], "checkpoint_sha256": file_sha256(best),
                       "recipe": recipe.to_audit_snapshot(), "engine_data": data_audit,
                       "metrics": evaluation["metrics"], "best_selection": "Ultralytics validation fitness; test excluded"})
            update({"status": "completed", "phase": "completed", "progress": 1, "metrics": evaluation["metrics"], "quality_status": "needs_review", "finished_at": now(), "eta_seconds": 0})
        return 0
    except (Exception, KeyboardInterrupt) as error:
        __import__("logging").exception("YOLO 实验失败或取消")
        update({"status": "cancelled" if cancelled() else "failed", "phase": "cancelled" if cancelled() else "failed", "error": str(error), "finished_at": now(), "eta_seconds": 0})
        return 1


def infer(config, source, job_dir, state, update, cancelled) -> int:
    from od_platform.common.video import VIDEO_SUFFIXES
    if Path(source).suffix.lower() in VIDEO_SUFFIXES:
        from od_platform.training.video_detection import infer_video
        return infer_video(config, source, update, cancelled)
    from ultralytics import YOLO
    if cancelled(): raise InterruptedError("已请求停止推理")
    with RunContext("detection_infer") as context:
        result = YOLO(config.model).predict(source=source, device=0, imgsz=config.imgsz, conf=config.threshold, iou=config.iou, verbose=False)[0]
        write_png(context.run_dir / "comparisons" / (Path(source).stem + ".png"), result.plot())
        write_json(context.run_dir / "predictions.json", {"names": result.names, "detections": result.summary()})
        write_json(context.run_dir / "odp_audit.json", {"config": config.model_dump(), "resolved_device": "cuda", "checkpoint_sha256": file_sha256(Path(config.model))})
        update({"status": "completed", "phase": "completed", "progress": 1, "run_dir": context.run_dir.relative_to(paths.ROOT_DIR).as_posix(), "finished_at": now(), "eta_seconds": 0})
    return 0
