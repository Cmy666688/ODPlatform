"""WSODD 检测实验配置,与二值分割配置分别校验。"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DetectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Literal["yolo"] = "yolo"
    data: str = "wsodd"
    model: Optional[str] = None
    architecture: Literal["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"] = "yolov8n.pt"
    device: str = "cuda"
    scan_backend: Literal["reference"] = "reference"  # 兼容远程设备白名单,YOLO 不使用扫描算子
    epochs: int = Field(default=50, ge=1, le=1000)
    batch_size_train: int = Field(default=16, ge=1, le=64)
    batch_size_test: int = Field(default=16, ge=1, le=64)
    lr: float = Field(default=0.001, gt=0, le=1)
    weight_decay: float = Field(default=0.0005, ge=0)
    optimizer: Literal["AdamW", "SGD"] = "AdamW"
    cos_lr: bool = False
    warmup_bias_lr: float = Field(default=0.1, ge=0, le=1)
    mosaic: float = Field(default=1.0, ge=0, le=1)
    scale: float = Field(default=0.5, ge=0, le=1)
    hsv_s: float = Field(default=0.7, ge=0, le=1)
    close_mosaic: int = Field(default=10, ge=0, le=1000)
    seed: int = 42
    imgsz: int = Field(default=640, ge=320, le=1280)
    num_threads: int = Field(default=4, ge=0, le=16)
    torch_threads: int = Field(default=4, ge=1)
    threshold: float = Field(default=0.25, gt=0, lt=1)
    iou: float = Field(default=0.7, gt=0, lt=1)
    max_train_batches: Optional[int] = Field(default=None, ge=1)
    max_eval_batches: Optional[int] = Field(default=None, ge=1)
