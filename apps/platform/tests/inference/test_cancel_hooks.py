"""cancel / hooks 单元测试: 取消信号与生命周期回调."""
from __future__ import annotations

import numpy as np
import pytest

from od_platform.inference import (
    CancelToken,
    FrameEvent,
    InferenceCancelled,
    InferHooks,
    ProgressEvent,
)


class TestCancelToken:
    def test_initial_state(self):
        t = CancelToken()
        assert t.is_cancelled() is False

    def test_cancel_is_idempotent(self):
        t = CancelToken()
        t.cancel()
        t.cancel()
        assert t.is_cancelled() is True

    def test_reset(self):
        t = CancelToken()
        t.cancel()
        t.reset()
        assert t.is_cancelled() is False

    def test_wait_returns_true_when_cancelled(self):
        t = CancelToken()
        t.cancel()
        assert t.wait(0.1) is True

    def test_wait_timeout_returns_false(self):
        assert CancelToken().wait(0.01) is False

    def test_exception_type(self):
        assert issubclass(InferenceCancelled, Exception)


class TestInferHooks:
    def _frame_evt(self) -> FrameEvent:
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        return FrameEvent(frame_idx=0, image=img, annotated=img,
                          n_detections=2, detections=None)

    def test_empty_hooks_do_nothing(self):
        h = InferHooks()
        h.fire_frame(self._frame_evt())
        h.fire_progress(ProgressEvent(1, None, 1.0, 10.0, 20.0, 3))
        h.fire_complete("r")
        h.fire_error(RuntimeError("x"))

    def test_callbacks_receive_events(self):
        seen = {}
        h = InferHooks(
            on_frame=lambda e: seen.setdefault("frame", e),
            on_progress=lambda e: seen.setdefault("progress", e),
            on_complete=lambda r: seen.setdefault("complete", r),
            on_error=lambda e: seen.setdefault("error", e),
        )
        h.fire_frame(self._frame_evt())
        h.fire_progress(ProgressEvent(30, 100, 2.0, 12.0, 18.0, 5))
        h.fire_complete("result")
        h.fire_error(ValueError("bad"))

        assert seen["frame"].n_detections == 2
        assert seen["progress"].total_frames == 100
        assert seen["progress"].frame_idx == 30
        assert seen["complete"] == "result"
        assert isinstance(seen["error"], ValueError)

    @pytest.mark.parametrize("firer,arg", [
        ("fire_frame", None),
        ("fire_progress", None),
        ("fire_complete", None),
        ("fire_error", None),
    ])
    def test_callback_exception_is_swallowed(self, firer, arg, caplog):
        def boom(_):
            raise RuntimeError("callback boom")

        h = InferHooks(on_frame=boom, on_progress=boom, on_complete=boom, on_error=boom)
        if firer == "fire_frame":
            h.fire_frame(self._frame_evt())
        elif firer == "fire_progress":
            h.fire_progress(ProgressEvent(1, None, 0.1, 1.0, 1.0, 0))
        elif firer == "fire_complete":
            h.fire_complete("x")
        else:
            h.fire_error(RuntimeError("original"))
        assert "回调异常" in caplog.text

    def test_default_progress_interval(self):
        assert InferHooks().progress_interval_frames == 30
