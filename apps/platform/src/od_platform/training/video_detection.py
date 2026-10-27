"""远程 YOLO 视频推理:流式逐帧结果、H.264 视频和可解释的类别摘要。"""
from pathlib import Path
import json
import subprocess

import cv2

from od_platform.common import paths
from od_platform.common.json_io import write_json
from od_platform.common.run_context import RunContext
from od_platform.common.semantic import write_png, file_sha256
from od_platform.common.video import inspect_video


def infer_video(config, source, update, cancelled, *, model_factory=None) -> int:
    import torch
    if config.device != "cuda" or not torch.cuda.is_available(): raise RuntimeError("视频模型推理必须使用远程 CUDA GPU")
    from ultralytics import YOLO
    from imageio_ffmpeg import get_ffmpeg_exe
    from od_platform.training.detection_experiment import now
    metadata = inspect_video(source); classes: dict[str, dict[str, float]] = {}; samples: list[dict] = []; count = 0
    with RunContext("detection_infer") as context:
        update({"run_dir": context.run_dir.relative_to(paths.ROOT_DIR).as_posix(), "phase": "inference", "media_type": "video", "total_frames": metadata["frames"]})
        raw = context.run_dir / "annotated.raw.mp4"; target = context.run_dir / "annotated.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter.fourcc(*"mp4v"), metadata["fps"], (metadata["width"], metadata["height"]))
        if not writer.isOpened(): raise RuntimeError("视频编码器不可用")
        results = None
        try:
            model = (model_factory or YOLO)(config.model)
            results = model.predict(source=source, stream=True, device=0, imgsz=config.imgsz, conf=config.threshold, iou=config.iou, max_det=100, verbose=False)
            with (context.run_dir / "frames.jsonl").open("w", encoding="utf-8") as stream:
                for result in results:
                    if cancelled(): raise InterruptedError("已请求停止视频推理")
                    count += 1
                    if count > 7200 or count / metadata["fps"] > 120 + 1/metadata["fps"]: raise ValueError("实际解码帧数或时长超过限制")
                    if result.orig_shape != (metadata["height"], metadata["width"]): raise ValueError("视频中途分辨率变化")
                    detections = result.summary(); per_frame: dict[str, int] = {}
                    for item in detections:
                        name = str(item["name"]); per_frame[name] = per_frame.get(name, 0) + 1
                        entry = classes.setdefault(name, {"frame_occurrences": 0, "max_simultaneous": 0, "max_confidence": 0.0})
                        entry["max_confidence"] = max(entry["max_confidence"], float(item["confidence"]))
                    for name, amount in per_frame.items():
                        classes[name]["frame_occurrences"] += 1
                        classes[name]["max_simultaneous"] = max(classes[name]["max_simultaneous"], amount)
                    timestamp = (count-1)/metadata["fps"]
                    stream.write(json.dumps({"frame": count, "seconds": timestamp, "detections": detections}, ensure_ascii=False) + "\n")
                    image = result.plot(); writer.write(image)
                    if len(samples) < 12 and (not samples or timestamp - samples[-1]["seconds"] >= max(1, metadata["duration_seconds"]/12)):
                        filename = f"frame_{count:06d}.png"; write_png(context.run_dir / "comparisons" / filename, image)
                        samples.append({"frame": count, "seconds": timestamp, "file": filename})
                    if count % 10 == 0: update({"processed_frames": count, "progress": min(0.9, 0.9*count/metadata["frames"])})
            writer.release()
            if not count: raise ValueError("视频未生成可用帧")
            if abs(count-metadata["frames"]) > 1: raise ValueError("视频截断或解码帧数与声明不符")
            if cancelled(): raise InterruptedError("已请求停止视频推理")
            update({"phase": "encoding", "processed_frames": count, "progress": 0.92})
            subprocess.run([get_ffmpeg_exe(), "-nostdin", "-y", "-loglevel", "error", "-i", str(raw), "-an", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", str(target)],
                           check=True, timeout=180, capture_output=True)
            if cancelled(): raise InterruptedError("已请求停止视频推理")
            summary = {"media_type": "video", "frames": count, "fps": metadata["fps"], "duration_seconds": count/metadata["fps"],
                       "class_summary": classes, "sampled_frames": samples, "threshold": config.threshold, "max_detections_per_frame": 100,
                       "note": "逐帧出现次数不代表独立目标数,没有跟踪 ID;无标注视频不能计算准确率,不能据此推断航速或碰撞风险"}
            write_json(context.run_dir / "video_summary.json", summary)
            write_json(context.run_dir / "odp_audit.json", {"config": config.model_dump(), "resolved_device": "cuda", "media_type": "video",
                                                         "checkpoint_sha256": file_sha256(Path(config.model)), "source_sha256": file_sha256(Path(source))})
            update({"status": "completed", "phase": "completed", "progress": 1, "finished_at": now(), "eta_seconds": 0})
        finally:
            writer.release()
            if results is not None and hasattr(results, "close"): results.close()
            raw.unlink(missing_ok=True)
    return 0
