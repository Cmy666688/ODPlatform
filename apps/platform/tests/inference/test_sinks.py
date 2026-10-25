"""sinks 单元测试: OutputSink / LocalFileSink / NullSink."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from od_platform.frame_source import SourceType
from od_platform.inference import LocalFileSink, NullSink, OutputSink


@dataclass
class FakeInfo:
    filename: str | None = None
    frame_index: int = 0
    fps: float | None = 30.0


@dataclass
class FakeFrame:
    image: np.ndarray
    info: FakeInfo


def _frame(name: str | None = "a.jpg", fps: float | None = 30.0) -> FakeFrame:
    return FakeFrame(np.zeros((48, 64, 3), dtype=np.uint8),
                     FakeInfo(filename=name, frame_index=0, fps=fps))


class TestAbstractSink:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            OutputSink()  # type: ignore[abstract]

    def test_subclass_must_implement_all(self):
        class Half(OutputSink):
            def open(self, output_dir, source_type):
                pass

        with pytest.raises(TypeError):
            Half()  # type: ignore[abstract]


class TestNullSink:
    def test_noop(self, tmp_path: Path):
        sink = NullSink()
        sink.open(tmp_path, SourceType.IMAGE_FOLDER)
        sink.write(_frame(), np.zeros((10, 10, 3), np.uint8))
        sink.close()
        sink.close()                      # 幂等
        assert list(tmp_path.iterdir()) == []


class TestLocalFileSink:
    def test_image_mode_writes_jpg_per_frame(self, tmp_path: Path):
        sink = LocalFileSink()
        sink.open(tmp_path, SourceType.IMAGE_FOLDER)
        sink.write(_frame("first.png"), np.zeros((10, 10, 3), np.uint8))
        sink.write(_frame("second.jpg"), np.zeros((10, 10, 3), np.uint8))
        sink.close()
        assert sorted(p.name for p in tmp_path.glob("*.jpg")) == ["first.jpg", "second.jpg"]
        assert sink.written == 2

    def test_image_mode_falls_back_to_frame_index_name(self, tmp_path: Path):
        sink = LocalFileSink()
        sink.open(tmp_path, SourceType.IMAGE)
        sink.write(_frame(None), np.zeros((10, 10, 3), np.uint8))
        sink.close()
        assert (tmp_path / "frame_000000.jpg").exists()

    def test_stream_mode_uses_video_writer(self, tmp_path: Path):
        sink = LocalFileSink()
        sink.open(tmp_path, SourceType.VIDEO)
        for _ in range(3):
            sink.write(_frame(), np.zeros((48, 64, 3), np.uint8))
        sink.close()
        assert sink.written == 3
        assert (tmp_path / "output.mp4").exists()

    def test_write_before_open_is_swallowed(self, tmp_path: Path, caplog):
        sink = LocalFileSink()
        sink.write(_frame(), np.zeros((10, 10, 3), np.uint8))
        assert "失败" in caplog.text

    def test_close_is_idempotent(self, tmp_path: Path):
        sink = LocalFileSink()
        sink.open(tmp_path, SourceType.VIDEO)
        sink.close()
        sink.close()

    def test_write_error_is_swallowed(self, tmp_path: Path, monkeypatch, caplog):
        import cv2
        sink = LocalFileSink()
        sink.open(tmp_path, SourceType.IMAGE_FOLDER)

        def boom(*a, **k):
            raise RuntimeError("disk gone")

        monkeypatch.setattr(cv2, "imwrite", boom)
        sink.write(_frame(), np.zeros((10, 10, 3), np.uint8))
        assert "LocalFileSink.write 失败" in caplog.text
