"""inference 测试共享夹具.

纪律:
  - 不依赖真实模型 / 网络下载: 用 FakeYOLO 替换 service.YOLO
  - 图片用 cv2 现场生成真实 jpg (帧源按真实文件读取)
  - 所有输出目录落在 tmp_path, 不污染项目 runs/
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest

import od_platform.inference.pipeline_config as pipeline_config_mod
import od_platform.inference.service as service_mod


# ============================================================================
# 假模型 (模拟 ultralytics Results 的最小接口)
# ============================================================================
class _FakeTensor:
    def __init__(self, arr) -> None:
        self._arr = np.asarray(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def int(self):
        return _FakeTensor(self._arr.astype(int))

    def tolist(self):
        return self._arr.tolist()


class FakeBoxes:
    def __init__(self, boxes, confs, classes) -> None:
        self.xyxy = _FakeTensor(boxes)
        self.conf = _FakeTensor(confs)
        self.cls = _FakeTensor(classes)

    def __len__(self) -> int:
        return len(self.cls.numpy())


class FakeResult:
    def __init__(self, boxes: FakeBoxes, speed: dict) -> None:
        self.boxes = boxes
        self.speed = speed

    def plot(self):
        return np.full((8, 8, 3), 7, dtype=np.uint8)


@dataclass
class FakeModelConfig:
    """FakeYOLO 的行为开关."""
    n_boxes: int = 2              # 每帧框数
    fail: bool = False            # 调用时抛异常


class _FakeYOLOSession:
    """一次测试内的假模型工厂记录 (可断言调用参数)."""

    def __init__(self, cfg: FakeModelConfig) -> None:
        self.cfg = cfg
        self.instances: list["FakeYOLO"] = []


class FakeYOLO:
    names = {0: "head", 1: "反光衣"}

    def __init__(self, path: str) -> None:
        self.path = path
        self.calls: list[list] = []

    def __call__(self, images, **kwargs):
        if self._fail:
            raise RuntimeError("fake model exploded")
        self.calls.append((images, kwargs))
        out = []
        for _ in images:
            if self._n_boxes == 0:
                boxes = FakeBoxes(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)))
            else:
                boxes = FakeBoxes(
                    [[4, 4, 40, 40]] * self._n_boxes,
                    [0.9] * self._n_boxes,
                    [0] + [1] * (self._n_boxes - 1),
                )
            out.append(FakeResult(boxes, {"preprocess": 1.0, "inference": 2.0,
                                          "postprocess": 0.5}))
        return out


# ============================================================================
# 夹具
# ============================================================================
@pytest.fixture
def make_images(tmp_path: Path) -> Callable[[int], Path]:
    """生成 n 张真实 jpg 的目录, 文件名 img_000.jpg ..."""
    import cv2

    def _make(n: int = 4) -> Path:
        d = tmp_path / "frames"
        d.mkdir(exist_ok=True)
        for i in range(n):
            img = np.full((96, 96, 3), 30, dtype=np.uint8)
            cv2.rectangle(img, (10 + i, 10), (60, 60), (200, 180, 160), -1)
            ok = cv2.imwrite(str(d / f"img_{i:03d}.jpg"), img)
            assert ok, "测试图片写入失败"
        return d

    return _make


@pytest.fixture
def isolated_paths(tmp_path: Path, monkeypatch) -> SimpleNamespace:
    """把 service/pipeline_config 的默认路径挪到 tmp (不读仓库真实配置)."""
    runs = tmp_path / "runs"
    runs.mkdir()
    configs = tmp_path / "configs"
    configs.mkdir()
    monkeypatch.setattr(service_mod, "RUNS_DIR", runs)
    monkeypatch.setattr(service_mod, "RUNTIME_CONFIGS_DIR", configs)
    monkeypatch.setattr(pipeline_config_mod, "RUNTIME_CONFIGS_DIR", configs)
    return SimpleNamespace(runs=runs, configs=configs)


@pytest.fixture
def install_yolo(monkeypatch) -> Callable[..., _FakeYOLOSession]:
    """把 service.YOLO 换成 FakeYOLO; 返回 session 便于断言."""
    def _install(n_boxes: int = 2, fail: bool = False) -> _FakeYOLOSession:
        session = _FakeYOLOSession(FakeModelConfig(n_boxes=n_boxes, fail=fail))

        class _Bound(FakeYOLO):
            def __init__(self, path: str) -> None:
                super().__init__(path)
                self._n_boxes = session.cfg.n_boxes
                self._fail = session.cfg.fail
                session.instances.append(self)

        monkeypatch.setattr(service_mod, "YOLO", _Bound)
        return session

    return _install


@pytest.fixture
def pipeline_yaml(tmp_path: Path) -> Path:
    """一份开启美化 + 类别映射的 infer_pipeline.yaml."""
    p = tmp_path / "infer_pipeline.yaml"
    p.write_text(
        """
frame_source:
  camera:
    width: 640
    height: 480
    fps: 25
visualization:
  enabled: true
  use_label_mapping: true
  label_mapping:
    head: 头部
  color_mapping:
    head: [0, 255, 0]
  default_color: [0, 0, 255]
  style:
    box_thickness: 3
""",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def base_cli_args(tmp_path: Path, make_images) -> dict:
    """指向临时图片目录的最小 cli_args."""
    frames = make_images(4)
    return {
        "source": str(frames),
        "conf": 0.25,
        "show": False,
        "save": True,
        "experiment_name": "unit",
    }
