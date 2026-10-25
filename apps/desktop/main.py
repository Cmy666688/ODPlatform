#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : main.py
# @Project   : ODPlatform
# @Function  : 最小桌面推理 demo (PySide6)
"""模型/输入选择、实时显示、进度、取消、关闭时异步收尾。

运行: python -m apps.desktop.main (仓库根),或 python apps/desktop/main.py。
引擎通过 editable 安装复用,桌面只实现 sink / hooks / cancel 三个接缝。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, QTimer, Qt, Slot
from PySide6.QtGui import QCloseEvent, QImage, QPixmap, QResizeEvent
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

# 直接执行脚本时只桥接桌面包所在的仓库根;引擎仍通过安装后的 od_platform 导入。
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from apps.desktop.infer_worker import InferWorker
from apps.desktop.qt_sink import QtSignalSink
from od_platform.common import paths
from od_platform.common.logging_utils import get_logger
from od_platform.inference import CancelToken, InferResult


class MainWindow(QMainWindow):
    """一次只允许一个推理任务,等待线程退出后才允许下次启动或关闭窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ODPlatform Desktop Demo")
        self.resize(1100, 760)
        self._thread: QThread | None = None
        self._worker: InferWorker | None = None
        self._sink: QtSignalSink | None = None
        self._token: CancelToken | None = None
        self._closing = False
        self._image: QImage | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        row_model = QHBoxLayout()
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["YOLO", "MixerCSeg"])
        self.backend_combo.currentTextChanged.connect(self._on_backend_changed)
        row_model.addWidget(self.backend_combo)
        row_model.addWidget(QLabel("模型:"))
        self.le_model = QLineEdit("yolov8n.pt")
        row_model.addWidget(self.le_model, 1)
        self.btn_model = QPushButton("选择权重...")
        self.btn_model.clicked.connect(self._browse_model)
        row_model.addWidget(self.btn_model)
        layout.addLayout(row_model)

        row_source = QHBoxLayout()
        row_source.addWidget(QLabel("输入:"))
        self.le_source = QLineEdit("0")
        self.le_source.setPlaceholderText("摄像头号、图片、目录、视频或网络流地址")
        row_source.addWidget(self.le_source, 1)
        self.btn_source = QPushButton("选择文件...")
        self.btn_source.clicked.connect(self._browse_source)
        row_source.addWidget(self.btn_source)
        self.btn_folder = QPushButton("选择目录...")
        self.btn_folder.clicked.connect(self._browse_folder)
        row_source.addWidget(self.btn_folder)
        layout.addLayout(row_source)

        row_buttons = QHBoxLayout()
        self.btn_start = QPushButton("开始")
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        row_buttons.addWidget(self.btn_start)
        row_buttons.addWidget(self.btn_stop)
        layout.addLayout(row_buttons)

        self.preview = QLabel("选择模型和输入后点击开始")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        layout.addWidget(self.preview, 1)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.statusBar().showMessage("就绪")

    @Slot()
    def _browse_model(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择模型权重", str(paths.TRAINED_MODELS_DIR),
                                                "模型文件 (*.pt *.pth *.onnx);;所有文件 (*)")
        if filename:
            self.le_model.setText(filename)

    @Slot()
    def _browse_source(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "选择图片或视频", str(paths.ROOT_DIR))
        if filename:
            self.le_source.setText(filename)

    @Slot()
    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "选择图片目录", str(paths.ROOT_DIR))
        if folder:
            self.le_source.setText(folder)

    @Slot(str)
    def _on_backend_changed(self, backend: str) -> None:
        if backend == "MixerCSeg":
            self.le_model.setText(str(paths.PRETRAINED_MODELS_DIR / "checkpoint_CrackMap.pth"))
            self.le_source.setText(str(paths.dataset_processed_dir("crackmap") / "images/test"))
        else:
            self.le_model.setText("yolov8n.pt")
            self.le_source.setText("0")

    def _set_running(self, running: bool) -> None:
        for widget in (self.btn_start, self.le_model, self.le_source,
                       self.btn_model, self.btn_source, self.btn_folder, self.backend_combo):
            widget.setEnabled(not running)
        self.btn_stop.setEnabled(running)
        self.progress.setVisible(running)

    @Slot()
    def _on_start(self) -> None:
        if self._thread is not None or self._closing:
            return
        model, source = self.le_model.text().strip(), self.le_source.text().strip()
        if not model or not source:
            self.statusBar().showMessage("请填写模型和输入源")
            return

        self._token = CancelToken()
        self._sink = QtSignalSink()
        self._sink.signals.frame_ready.connect(self._on_frame, Qt.ConnectionType.QueuedConnection)
        self._worker = InferWorker(model, source, self._sink, self._token,
                                   backend=self.backend_combo.currentText().lower())
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.start_infer)
        self._worker.progress_changed.connect(self._on_progress, Qt.ConnectionType.QueuedConnection)
        self._worker.inference_finished.connect(self._on_finished, Qt.ConnectionType.QueuedConnection)
        self._worker.inference_finished.connect(self._thread.quit, Qt.ConnectionType.DirectConnection)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._set_running(True)
        self.progress.setRange(0, 0)
        self.statusBar().showMessage("正在加载模型并推理...")
        self._thread.start()

    @Slot()
    def _on_stop(self) -> None:
        # 直接设置 threading.Event,不能排队到被推理阻塞的 worker 事件循环。
        if self._token is not None:
            self._token.cancel()
            self.btn_stop.setEnabled(False)
            self.statusBar().showMessage("正在停止,等待当前操作完成...")

    @Slot(object)
    def _on_frame(self, frame) -> None:
        if self._closing or not isinstance(frame, np.ndarray):
            return
        h, w = frame.shape[:2]
        self._image = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888).copy()
        self._show_image()

    def _show_image(self) -> None:
        if self._image is not None:
            pixmap = QPixmap.fromImage(self._image)
            self.preview.setPixmap(pixmap.scaled(self.preview.size(),
                                                Qt.AspectRatioMode.KeepAspectRatio,
                                                Qt.TransformationMode.SmoothTransformation))

    @Slot(int, int, float)
    def _on_progress(self, frame_idx: int, total: int, fps: float) -> None:
        if total > 0:
            self.progress.setRange(0, total)
            self.progress.setValue(frame_idx)
        if not self._closing and self.btn_stop.isEnabled():
            self.statusBar().showMessage(f"已处理 {frame_idx} 帧 | {fps:.1f} FPS")

    @Slot(object)
    def _on_finished(self, result: InferResult) -> None:
        if not result.success:
            self.statusBar().showMessage(f"推理失败: {result.error}")
        elif self._token is not None and self._token.is_cancelled():
            self.statusBar().showMessage("已停止")
        else:
            self.statusBar().showMessage(f"完成 | {result.stats.get('frames', 0)} 帧 | 审计: {result.audit_path}")

    @Slot()
    def _on_thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self._sink = None
        self._token = None
        self._set_running(False)
        if self._closing:
            QTimer.singleShot(0, self.close)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._show_image()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is not None:
            self._closing = True
            self._on_stop()
            event.ignore()                      # 保持事件循环,线程退出后再关闭
            return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    get_logger(paths.LOGGING_DIR, log_type="desktop")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
