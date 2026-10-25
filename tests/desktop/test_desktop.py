"""桌面接缝回归:跨线程图像所有权、取消、失败、重跑和关窗收尾。

使用 offscreen Qt + 假推理服务,不下载模型、不启动摄像头、不写真实 runs/。
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication

from apps.desktop import infer_worker as worker_mod
from apps.desktop.main import MainWindow
from apps.desktop.qt_sink import QtSignalSink
from od_platform.frame_source import SourceType
from od_platform.inference import InferResult, ProgressEvent


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _wait(qapp, predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()
    assert predicate(), "等待 Qt 信号/线程收尾超时"


@pytest.fixture
def window(qapp):
    win = MainWindow()
    win.le_source.setText("test-images")
    yield win
    if win._thread is not None:
        win._on_stop()
        _wait(qapp, lambda: win._thread is None)
    win.close()
    qapp.processEvents()


def test_sink_copies_frame_and_close_is_idempotent(qapp, tmp_path):
    sink = QtSignalSink()
    seen = []
    sink.signals.frame_ready.connect(seen.append)
    sink.open(tmp_path, SourceType.IMAGE)
    image = np.zeros((12, 12, 3), dtype=np.uint8)[:, ::2]
    sink.write(None, image)
    image[:] = 255
    assert seen[0].flags["C_CONTIGUOUS"]
    assert not seen[0].any()                    # 跨线程帧不跟着原 buffer 变化
    assert sink.write_count == 1
    sink.close(); sink.close()
    sink.write(None, image)
    assert sink.write_count == 1


def test_sink_bad_frame_does_not_escape(qapp, tmp_path):
    sink = QtSignalSink()
    sink.open(tmp_path, SourceType.IMAGE)
    sink.write(None, np.zeros((10, 10), dtype=np.uint8))
    assert sink.write_count == 0
    sink.close()


def test_window_receives_frame_and_can_run_again(window, qapp, monkeypatch, tmp_path):
    threads = []
    def fake_infer(**kwargs):
        threads.append(QThread.currentThread())
        sink = kwargs["output_sink"]
        sink.open(tmp_path, SourceType.IMAGE)
        sink.write(None, np.full((24, 32, 3), 80, dtype=np.uint8))
        sink.close()
        kwargs["hooks"].fire_progress(ProgressEvent(1, 1, 0.1, 10.0, 20.0, 1))
        return InferResult(True, tmp_path, stats={"frames": 1}, audit_path=tmp_path / "audit.json")
    monkeypatch.setattr(worker_mod, "infer_yolo", fake_infer)
    for _ in range(2):
        window._on_start()
        _wait(qapp, lambda: window._thread is None)
        assert window.btn_start.isEnabled()
        assert window._image is not None and not window._image.isNull()
        assert "完成" in window.statusBar().currentMessage()
    assert len(threads) == 2
    assert all(thread != qapp.thread() for thread in threads)


def test_stop_reaches_busy_worker(window, qapp, monkeypatch, tmp_path):
    started = threading.Event()
    cancelled = threading.Event()
    def fake_infer(**kwargs):
        started.set()
        if kwargs["cancel_token"].wait(3.0):
            cancelled.set()
        return InferResult(True, tmp_path)
    monkeypatch.setattr(worker_mod, "infer_yolo", fake_infer)
    window._on_start()
    _wait(qapp, started.is_set)
    window._on_stop()
    _wait(qapp, lambda: window._thread is None)
    assert cancelled.is_set()
    assert window.statusBar().currentMessage() == "已停止"


@pytest.mark.parametrize("raises", [False, True])
def test_failure_still_releases_worker(window, qapp, monkeypatch, tmp_path, raises):
    def fake_infer(**kwargs):
        if raises:
            raise RuntimeError("fake failure")
        return InferResult(False, tmp_path, error="fake failure")
    monkeypatch.setattr(worker_mod, "infer_yolo", fake_infer)
    window._on_start()
    _wait(qapp, lambda: window._thread is None)
    assert window.btn_start.isEnabled()
    assert "fake failure" in window.statusBar().currentMessage()


def test_close_cancels_then_waits_for_thread(window, qapp, monkeypatch, tmp_path):
    started = threading.Event()
    def fake_infer(**kwargs):
        started.set()
        kwargs["cancel_token"].wait(3.0)
        return InferResult(True, tmp_path)
    monkeypatch.setattr(worker_mod, "infer_yolo", fake_infer)
    window.show()
    window._on_start()
    _wait(qapp, started.is_set)
    window.close()
    _wait(qapp, lambda: window._thread is None and not window.isVisible())
    assert window._closing


def test_stop_before_worker_starts_is_preserved(window, qapp, monkeypatch, tmp_path):
    seen = []
    def fake_infer(**kwargs):
        kwargs["cancel_token"].wait(1.0)
        seen.append(kwargs["cancel_token"].is_cancelled())
        return InferResult(True, tmp_path)
    monkeypatch.setattr(worker_mod, "infer_yolo", fake_infer)
    window._on_start()
    window._on_stop()
    _wait(qapp, lambda: window._thread is None)
    assert seen == [True]


def test_backend_selection_uses_crackmap_defaults(window):
    window.backend_combo.setCurrentText("MixerCSeg")
    assert window.le_model.text().endswith("checkpoint_CrackMap.pth")
    assert "crackmap" in window.le_source.text()
    window.backend_combo.setCurrentText("YOLO")
    assert window.le_model.text() == "yolov8n.pt"
