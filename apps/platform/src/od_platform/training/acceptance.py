"""独立 test 集验收:原图尺寸像素指标、空预测基线、逐图误差和训练曲线。"""
from __future__ import annotations

from pathlib import Path
import csv

import cv2
import numpy as np

from od_platform.common.json_io import write_json
from od_platform.common.semantic import file_sha256, load_semantic_dataset, paired_samples, read_bgr, read_binary_mask, write_png
from od_platform.runtime_config.mixercseg import MixerCSegConfig


def render_curves(history: list, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    axes[0].plot(epochs, [r["train_loss"] for r in history], label="Train", color="#167d68")
    axes[0].plot(epochs, [r["validation"]["loss"] for r in history], label="Validation", color="#e0a23d")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    for key, label in [("F1", "F1"), ("foreground_IoU", "Foreground IoU"), ("mIoU", "mIoU")]:
        axes[1].plot(epochs, [r["validation"][key] for r in history], label=label)
    axes[1].set_title("Validation metrics"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1); axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.15); ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(output, dpi=160); plt.close(fig)


def evaluate_checkpoint(config: MixerCSegConfig, data_yaml: Path, checkpoint: Path,
                        output: Path, *, history=None, min_f1=None,
                        on_progress=None, should_cancel=None) -> dict:
    import torch
    from od_platform.backends.mixercseg.data import image_tensor
    from od_platform.backends.mixercseg.model import build_model
    from od_platform.backends.mixercseg.metrics import BinaryMetrics
    from od_platform.visualization import MaskStyle, MaskVisualizer
    from od_platform.data_validation.semantic import validate_semantic_dataset
    output.mkdir(parents=True, exist_ok=True)
    quality = validate_semantic_dataset(data_yaml, write_report=False)
    if quality.overall_severity == "ERROR":
        raise ValueError("测试前数据质检 ERROR,拒绝生成验收结论")
    document, root = load_semantic_dataset(data_yaml)
    model, _, device = build_model(config, checkpoint=str(checkpoint))
    model.eval()
    samples = paired_samples(root, "test")
    if not samples:
        raise ValueError("独立测试集为空")
    metrics, baseline = BinaryMetrics(), BinaryMetrics()
    visualizer = MaskVisualizer(MaskStyle(threshold=config.threshold))
    rows = []
    with torch.no_grad():
        for index, (image_path, mask_path) in enumerate(samples):
            if should_cancel is not None and should_cancel():
                raise InterruptedError("用户请求停止测试验收")
            image = read_bgr(image_path); truth = read_binary_mask(mask_path) > 0
            inputs = image_tensor(image, config.load_width, config.load_height).unsqueeze(0).to(device)
            probability = model(inputs).sigmoid().cpu().numpy()[0, 0]
            probability = cv2.resize(probability, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            single = BinaryMetrics(); single.update(probability, truth, config.threshold)
            metrics.update(probability, truth, config.threshold)
            baseline.update(np.zeros_like(probability), truth, config.threshold)
            name = f"{index:03d}-{image_path.stem}.png"
            write_png(output / "masks" / name, (probability >= config.threshold).astype(np.uint8) * 255)
            write_png(output / "comparisons" / name, visualizer.comparison(image, probability, truth))
            rows.append({"filename": image_path.name, **single.to_dict(),
                         "tp": single.tp, "fp": single.fp, "fn": single.fn, "tn": single.tn,
                         "mask": f"masks/{name}", "comparison": f"comparisons/{name}"})
            if on_progress is not None:
                on_progress({"phase": "testing", "batch": index + 1, "batches": len(samples), "history": history or []})
    values = metrics.to_dict()
    report = {"split": "test", "images": len(rows), "threshold": config.threshold,
              "metric_policy": "global pixel confusion; predictions resized to native mask dimensions",
              "checkpoint_sha256": file_sha256(checkpoint), "data_fingerprint": document["data_fingerprint"],
              "metrics": values, "confusion": {"tp": metrics.tp, "fp": metrics.fp, "fn": metrics.fn, "tn": metrics.tn},
              "empty_baseline": baseline.to_dict(), "per_image": rows,
              "min_f1": min_f1, "quality_status": "needs_review" if min_f1 is None else "passed" if values["F1"] >= min_f1 else "failed",
              "quality_note": "未指定业务精度目标;仅报告实际效果,不宣称达到论文或生产精度",
              "dataset_warning": "; ".join(row.summary for row in quality.results if row.severity == "WARNING") or "无 WARNING"}
    write_json(output / "report.json", report)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if history:
        render_curves(history, output / "training_curves.png")
    (output / "report.md").write_text(
        f"# {config.data} 独立测试集验收\n\n"
        f"测试图片: {len(rows)};固定阈值: {config.threshold};原图尺寸像素统计。\n\n"
        + "\n".join(f"- {k}: {v:.6f}" for k, v in values.items())
        + "\n\n" + report["quality_note"] + "。\n\n" + report["dataset_warning"] + "。\n", encoding="utf-8")
    return report
