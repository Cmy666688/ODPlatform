"""受限视频探测。解码属于数据预处理,模型推理只能在远程 CUDA 执行。"""
from pathlib import Path
import math

import cv2

MAX_VIDEO_BYTES = 200 * 1024**2
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


def inspect_video(source):
    source = Path(source)
    if source.suffix.lower() not in VIDEO_SUFFIXES or source.stat().st_size > MAX_VIDEO_BYTES:
        raise ValueError("请选择最大 200 MB 的 MP4、AVI、MOV 或 MKV 视频")
    capture = cv2.VideoCapture(str(source))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS); frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or not 0 < fps <= 60 or not math.isfinite(frames) or not 1 <= frames <= 7200:
            raise ValueError("视频帧率/帧数不可读取或超过限制（60 FPS / 7200 帧）")
        if frames / fps > 120: raise ValueError("视频最长 120 秒")
        ok, frame = capture.read()
        if not ok or frame is None: raise ValueError("视频首帧无法解码")
        height, width = frame.shape[:2]
        if width * height > 1920 * 1080 or min(width, height) < 2: raise ValueError("视频分辨率最高为 1920×1080 像素总量")
        return {"fps": fps, "frames": int(frames), "width": width, "height": height, "duration_seconds": frames/fps}
    finally: capture.release()
