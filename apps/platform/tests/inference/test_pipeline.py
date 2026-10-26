"""pipeline 单元测试: 多线程流水线 (帧源用真实图片夹, 模型用 FakeProcessor)."""
from __future__ import annotations

from pathlib import Path

import pytest

from od_platform.frame_source import SourceType
from od_platform.inference import CancelToken, InferHooks, OutputSink
from od_platform.inference.pipeline import ThreadedPipeline
from od_platform.inference.service import InferStats


class FakeProcessor:
    """只有 infer_batch / draw 两个方法的假帧处理器."""

    def __init__(self) -> None:
        self.infer_batches = 0

    def infer_batch(self, images):
        self.infer_batches += 1
        results = [object() for _ in images]
        labels = [["head"] for _ in images]
        counts = [1 for _ in images]
        return results, labels, counts, 0.002

    def draw(self, image, result, labels, n):
        out = image.copy()
        out[:, :, 0] = 200
        return out


class RecordingSink(OutputSink):
    def __init__(self) -> None:
        self.opened = 0
        self.closed = 0
        self.frames = 0
        self.source_type = None

    def open(self, output_dir: Path, source_type: SourceType) -> None:
        self.opened += 1
        self.source_type = source_type

    def write(self, frame, annotated) -> None:
        self.frames += 1

    def close(self) -> None:
        self.closed += 1


def _build(image_dir: Path, *, sink=None, batch=2, save=True, warmup=0,
           hooks=None, cancel=None, show=False) -> tuple[ThreadedPipeline, InferStats]:
    stats = InferStats()
    pipe = ThreadedPipeline(
        processor=FakeProcessor(),
        source=str(image_dir),
        camera_config=None,
        output_dir=image_dir.parent / "out",
        output_sink=sink or RecordingSink(),
        batch_size=batch,
        save=save,
        show=show,
        show_info=False,
        window_name="test",
        warmup_frames=warmup,
        hooks=hooks,
        cancel_token=cancel,
    )
    return pipe, stats


class TestThreadedPipeline:
    def test_processes_all_frames_and_closes_sink(self, make_images):
        images = make_images(5)
        sink = RecordingSink()
        pipe, stats = _build(images, sink=sink, batch=2)
        interrupted = pipe.run(stats)

        assert interrupted is False
        assert stats.frames == 5
        assert sink.opened == 1
        assert sink.closed == 1
        assert sink.frames == 5
        assert sink.source_type == SourceType.IMAGE_FOLDER

    def test_warmup_drops_leading_frames(self, make_images):
        images = make_images(6)
        pipe, stats = _build(images, batch=2, warmup=2)
        pipe.run(stats)
        assert stats.frames == 4

    def test_save_true_writes_local_jpg(self, make_images, tmp_path):
        images = make_images(3)
        from od_platform.inference import LocalFileSink
        out = tmp_path / "out3"
        out.mkdir()
        sink = LocalFileSink()
        stats = InferStats()
        pipe = ThreadedPipeline(
            processor=FakeProcessor(), source=str(images), camera_config=None,
            output_dir=out, output_sink=sink, batch_size=2, save=True,
            show=False, show_info=False, window_name="t", warmup_frames=0,
        )
        pipe.run(stats)
        assert len(list(out.glob("*.jpg"))) == 3

    def test_save_false_does_not_write_files(self, make_images, tmp_path):
        images = make_images(3)
        out = tmp_path / "out_nosave"
        out.mkdir()
        stats = InferStats()
        pipe = ThreadedPipeline(
            processor=FakeProcessor(), source=str(images), camera_config=None,
            output_dir=out, output_sink=RecordingSink(), batch_size=2, save=False,
            show=False, show_info=False, window_name="t", warmup_frames=0,
        )
        pipe.run(stats)
        assert list(out.iterdir()) == []
        assert stats.frames == 3

    def test_progress_hook_fires_per_interval(self, make_images):
        images = make_images(4)
        events = []
        hooks = InferHooks(on_progress=events.append, progress_interval_frames=2)
        pipe, stats = _build(images, batch=2, hooks=hooks)
        pipe.run(stats)
        assert [e.frame_idx for e in events] == [2, 4]
        assert events[-1].detections_total == 4

    def test_frame_hook_fires_for_each_frame(self, make_images):
        images = make_images(3)
        events = []
        hooks = InferHooks(on_frame=events.append)
        pipe, stats = _build(images, batch=2, hooks=hooks)
        pipe.run(stats)
        assert len(events) == 3
        assert [e.frame_idx for e in events] == [0, 1, 2]
        assert all(e.annotated.shape == e.image.shape for e in events)

    def test_pre_cancelled_token_stops_before_any_batch(self, make_images):
        images = make_images(4)
        token = CancelToken()
        token.cancel()
        sink = RecordingSink()
        pipe, stats = _build(images, sink=sink, cancel=token)
        interrupted = pipe.run(stats)
        assert interrupted is True
        assert stats.frames == 0
        assert sink.frames == 0

    def test_stats_multidim_fps_filled(self, make_images):
        images = make_images(4)
        pipe, stats = _build(images, batch=2)
        pipe.run(stats)
        assert stats.capture_fps > 0
        assert stats.render_fps > 0
        assert stats.loop_fps > 0
        assert stats.avg_latency_ms > 0

    def test_batch_size_at_least_one(self, make_images):
        images = make_images(2)
        pipe, stats = _build(images, batch=0)
        pipe.run(stats)
        assert stats.frames == 2


class TestReader:
    def test_empty_source_finishes_without_frames(self, tmp_path):
        """帧源本身不产帧 (空目录): 流水线应正常收尾而不是卡死."""
        empty = tmp_path / "empty"
        empty.mkdir()
        stats = InferStats()
        pipe = ThreadedPipeline(
            processor=FakeProcessor(), source=str(empty),
            camera_config=None, output_dir=tmp_path / "out",
            output_sink=RecordingSink(), batch_size=2, save=False,
            show=False, show_info=False, window_name="t", warmup_frames=0,
        )
        assert pipe.run(stats) is False
        assert stats.frames == 0


def test_incomplete_model_batch_fails_instead_of_silently_dropping_frames(make_images):
    """模型返回条数不足时明确失败,不能把缺少结果的帧静默算作完成。"""
    pipe, stats = _build(make_images(3), batch=2)
    class IncompleteProcessor(FakeProcessor):
        def infer_batch(self, images):
            results, labels, counts, dt = super().infer_batch(images)
            return results[:-1], labels, counts, dt
    pipe.proc = IncompleteProcessor()
    with pytest.raises(ValueError, match="zip"):
        pipe.run(stats)


def test_semantic_renderer_receives_frame_metadata(make_images):
    pipe, stats = _build(make_images(2), batch=1)
    seen = []
    class MetadataProcessor(FakeProcessor):
        def draw_frame(self, frame, result, labels, n):
            seen.append(frame.info.filename)
            return self.draw(frame.image, result, labels, n)
    pipe.proc = MetadataProcessor()
    pipe.run(stats)
    assert len(seen) == stats.frames == 2
    assert all(name.endswith(".jpg") for name in seen)


def test_saved_large_frame_finishes_rendering_before_sink_closes(make_images):
    import time
    sink = RecordingSink(); pipe, stats = _build(make_images(1), sink=sink, save=True)
    class SlowProcessor(FakeProcessor):
        def draw(self, image, result, labels, n):
            time.sleep(3.2)
            assert sink.closed == 0
            return super().draw(image, result, labels, n)
    pipe.proc = SlowProcessor()
    assert pipe.run(stats) is False
    assert stats.frames == sink.frames == 1 and sink.closed == 1
