#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : mixercseg_service.py
# @Project   : ODPlatform
# @Function  : MixerCSeg 接入 D7 帧源/线程/输出/回调/取消与掩膜可视化
"""独立模型后端,复用 D7 流水线。原始 mask、对比图与审计同属本次运行。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import logging
import re
import time

import cv2
import numpy as np

from od_platform.common import paths, refs
from od_platform.common.run_context import RunContext
from od_platform.common.semantic import file_sha256, load_semantic_dataset, read_binary_mask, write_png
from od_platform.runtime_config.mixercseg import MixerCSegConfig, UPSTREAM_COMMIT
from od_platform.visualization import MaskStyle, MaskVisualizer

from .cancel import CancelToken
from .hooks import InferHooks
from .pipeline import ThreadedPipeline
from .pipeline_config import load_pipeline_config
from .service import InferResult, InferStats
from .sinks import LocalFileSink, NullSink, OutputSink

logger = logging.getLogger(__name__)


@dataclass
class SemanticPrediction:
    probabilities: np.ndarray
    sequence: int
    speed: dict


class MixerProcessor:
    def __init__(self, model, device: str, config: MixerCSegConfig, visualizer: MaskVisualizer,
                 output_dir: Path | None = None, truth_dir: Path | None = None, beautify=True):
        self.model, self.device, self.config = model, device, config
        self.visualizer, self.output_dir, self.truth_dir = visualizer, output_dir, truth_dir
        self.sequence = 0
        self.masks_saved = 0
        self.rendered = 0
        self.beautify = beautify

    def infer_batch(self, images: list):
        import torch
        from od_platform.backends.mixercseg.data import image_tensor
        start = time.perf_counter()
        batch = torch.stack([image_tensor(image, self.config.load_width, self.config.load_height)
                             for image in images]).to(self.device)
        prepared = time.perf_counter()
        with torch.no_grad():
            probabilities = self.model(batch).sigmoid().cpu().numpy()[:, 0]
        completed = time.perf_counter()
        predictions = []; labels = []; counts = []
        for image, probability in zip(images, probabilities, strict=True):
            h, w = image.shape[:2]
            probability = cv2.resize(probability, (w, h), interpolation=cv2.INTER_LINEAR)
            if not np.isfinite(probability).all():
                raise RuntimeError("模型预测存在 NaN/Inf")
            speed = {"preprocess": (prepared - start) * 1000 / len(images),
                     "inference": (completed - prepared) * 1000 / len(images), "postprocess": 0.0}
            predictions.append(SemanticPrediction(probability, self.sequence, speed))
            positive = bool(np.any(probability >= self.config.threshold))
            counts.append(int(positive)); labels.append(["crack"] if positive else [])
            self.sequence += 1
        return predictions, labels, counts, completed - prepared

    def draw(self, image, result: SemanticPrediction, labels, n):
        return self.visualizer.draw(image, result.probabilities) if self.beautify else image.copy()

    def draw_frame(self, frame, result: SemanticPrediction, labels, n):
        annotated = self.draw(frame.image, result, labels, n)
        if self.output_dir is not None:
            stem = Path(frame.info.filename or "frame").stem
            safe_stem = re.sub(r"[^\w.() -]", "_", stem)
            name = f"{result.sequence:06d}-{safe_stem}.png"
            write_png(self.output_dir / "masks" / name,
                      (result.probabilities >= self.config.threshold).astype(np.uint8) * 255)
            truth = None
            if self.truth_dir is not None and (self.truth_dir / (stem + ".png")).is_file():
                truth = read_binary_mask(self.truth_dir / (stem + ".png"))
            if self.beautify:
                comparison = self.visualizer.comparison(frame.image, result.probabilities, truth)
                write_png(self.output_dir / "comparisons" / name, comparison)
            self.masks_saved += 1
        self.rendered += 1
        return annotated


def infer_mixercseg(config: MixerCSegConfig, source: str | None = None, *,
                    pipeline_yaml=None, save=True, show=False, show_info=True, beautify=True, warmup_frames=0,
                    output_sink: OutputSink | None = None, hooks: InferHooks | None = None,
                    cancel_token: CancelToken | None = None, truth_dir: Path | None = None) -> InferResult:
    hooks = hooks or InferHooks()
    start = time.perf_counter()
    output_dir = Path("unknown")
    try:
        from od_platform.backends.mixercseg.model import build_model, resolve_checkpoint
        data_yaml = refs.resolve_dataset_yaml(config.data)
        root = load_semantic_dataset(data_yaml)[1] if data_yaml.exists() else None
        if source is None:
            if root is None:
                raise FileNotFoundError("默认测试集不存在,请先 odp-prepare-cracks 或传 --source")
            source = str(root / "images" / "test")
        if root is not None and truth_dir is None and not source.isdigit() and "://" not in source:
            source_path = Path(source)
            if source_path.resolve().parent == (root / "images").resolve():
                truth_dir = root / "masks" / source_path.name
        if config.model is None:
            raise ValueError("MixerCSeg 推理必须指定 .pth 权重")
        model, _criterion, device = build_model(config, checkpoint=config.model)
        model.eval()
        pipe = load_pipeline_config(pipeline_yaml or paths.RUNTIME_CONFIGS_DIR / "mixercseg_pipeline.yaml")
        style_args = dict(pipe.mask_style)
        style_args["threshold"] = config.threshold
        visualizer = MaskVisualizer(MaskStyle(**style_args))
        with RunContext("semantic_infer") as run:
            output_dir = run.run_dir
            stats = InferStats()
            processor = MixerProcessor(model, device, config, visualizer,
                                       output_dir if save else None, truth_dir, beautify and pipe.viz_enabled)
            sink = output_sink if output_sink is not None else LocalFileSink() if save else NullSink()
            pipeline = ThreadedPipeline(processor=processor, source=source,
                                        camera_config=pipe.build_camera_config(), output_dir=output_dir,
                                        output_sink=sink, batch_size=config.batch_size_test, save=save,
                                        show=show, show_info=show_info, window_name="MixerCSeg",
                                        warmup_frames=warmup_frames, hooks=hooks, cancel_token=cancel_token)
            interrupted = pipeline.run(stats)
            if not interrupted and (stats.frames == 0 or processor.rendered != stats.frames):
                raise RuntimeError(f"输入为空或存在绘制失败: 推理 {stats.frames} 帧,绘制 {processor.rendered} 帧")
            values = stats.to_dict()
            values["positive_frames"] = values.pop("detections")  # 语义前景不是独立实例数
            values["masks_saved"] = processor.masks_saved
            audit_path = run.artifact_path("odp_audit.json")
            audit = {"backend": "mixercseg", "task": "semantic_segment", "upstream_commit": UPSTREAM_COMMIT,
                     "config": config.model_dump(), "resolved_device": device,
                     "checkpoint_sha256": file_sha256(resolve_checkpoint(config.model)),
                     "source": source, "truth_dir": str(truth_dir) if truth_dir else None,
                     "pipeline": pipe.to_audit(), "stats": values, "interrupted": interrupted,
                     "probability_policy": "sigmoid(logits);fixed threshold;no per-image max normalization"}
            audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
            result = InferResult(True, output_dir, values, time.perf_counter() - start, save,
                                 audit_path=audit_path)
        hooks.fire_complete(result)
        return result
    except Exception as e:
        logger.exception("MixerCSeg 推理失败: %s", e)
        hooks.fire_error(e)
        return InferResult(False, output_dir, error=str(e), infer_time=time.perf_counter() - start)
