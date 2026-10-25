"""MixerCSeg CLI 适配。训练/推理仍由各自 service 执行。"""
from __future__ import annotations

import logging

from od_platform.common import paths, refs
from od_platform.common.logging_utils import get_logger
from od_platform.common.run_context import RunContext
from od_platform.runtime_config.mixercseg import load_mixercseg_config

logger = logging.getLogger(__name__)


def train_main(args) -> int:
    get_logger(paths.LOGGING_DIR, log_type="train_mixercseg")
    try:
        from od_platform.training.mixercseg_service import train_mixercseg
        if args.skip_validate:
            raise ValueError("MixerCSeg 始终核验图像和掩膜;--skip-validate 仅适用于 YOLO")
        if args.imgsz is not None and args.imgsz != 512:
            raise ValueError("原版 MixerCSeg 配方要求 --imgsz 512")
        batch = int(args.batch) if args.batch is not None else None
        overrides = {"data": args.data, "model": args.model, "epochs": args.epochs,
                     "batch_size_train": batch, "device": args.device,
                     "experiment_name": args.experiment_name,
                     "max_train_batches": args.smoke_batches, "max_eval_batches": args.smoke_batches}
        reference = "mixercseg_train" if args.config == "train" else args.config
        config = load_mixercseg_config(reference, overrides)
        with RunContext("training") as run:
            result = train_mixercseg(config, refs.resolve_dataset_yaml(config.data), run,
                                     archive=not args.no_archive)
        logger.info("MixerCSeg 训练结果: success=%s | %s | %s", result.success, result.save_dir, result.message)
        return 0 if result.success else 1
    except Exception as e:
        logger.exception("MixerCSeg 配置/训练失败: %s", e)
        return 1


def infer_main(args) -> int:
    get_logger(paths.LOGGING_DIR, log_type="infer_mixercseg")
    try:
        from od_platform.inference.mixercseg_service import infer_mixercseg
        if args.iou is not None or args.classes is not None or args.max_det is not None:
            raise ValueError("MixerCSeg 为二值语义分割,不接受 YOLO 的 --iou/--classes/--max-det")
        if args.imgsz is not None and args.imgsz != 512:
            raise ValueError("原版 MixerCSeg 配方要求 --imgsz 512")
        config = load_mixercseg_config(args.yaml or "mixercseg_infer",
                                      {"model": args.model, "device": args.device,
                                       "threshold": args.conf, "batch_size_test": args.batch,
                                       "experiment_name": args.experiment_name})
        result = infer_mixercseg(config, args.source, pipeline_yaml=args.pipeline_yaml,
                                 save=args.save is not False, show=args.show is True,
                                 show_info=not args.no_hud, beautify=not args.no_viz,
                                 warmup_frames=args.warmup)
        logger.info("MixerCSeg 推理结果: success=%s | %s | %s", result.success, result.output_dir, result.error)
        return 0 if result.success else 1
    except Exception as e:
        logger.exception("MixerCSeg 配置/推理失败: %s", e)
        return 1
