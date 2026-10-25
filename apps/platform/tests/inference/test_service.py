"""service 单元测试: InferService 八阶段编排 + 接缝 + 审计快照."""
from __future__ import annotations

import json

import numpy as np
import pytest

import od_platform.inference.service as svc
from od_platform.inference import (
    CancelToken,
    InferHooks,
    InferStats,
    OutputSink,
    infer_yolo,
)
from od_platform.visualization import BeautifyVisualizer, DrawStyle


class RecordingSink(OutputSink):
    def __init__(self) -> None:
        self.opened = 0
        self.closed = 0
        self.frames = 0

    def open(self, output_dir, source_type) -> None:
        self.opened += 1

    def write(self, frame, annotated) -> None:
        self.frames += 1

    def close(self) -> None:
        self.closed += 1


# ============================================================================
# 端到端 (假模型 + 真实图片夹)
# ============================================================================
class TestPredictEndToEnd:
    def test_success_writes_images_and_audit(self, install_yolo, isolated_paths,
                                             pipeline_yaml, base_cli_args):
        install_yolo(n_boxes=2)
        result = infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=base_cli_args)

        assert result.success is True, result.error
        assert result.stats["frames"] == 4
        assert result.stats["detections"] == 8
        assert result.stats["per_class"] == {"head": 4, "反光衣": 4}
        assert result.output_dir.parent == isolated_paths.runs / "detect_infer"
        assert len(list(result.output_dir.glob("*.jpg"))) == 4
        assert result.saved is True
        assert result.infer_time is not None and result.infer_time >= 0

        audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
        assert audit["mode"] == "infer"
        assert audit["pipeline"]["viz_enabled"] is True
        assert audit["pipeline"]["label_mapping_n"] == 1
        assert audit["stats"]["frames"] == 4
        assert audit["result_summary"]["saved"] is True
        assert audit["result_summary"]["beautified"] is True

    def test_predict_kwargs_only_predict_whitelist(self, install_yolo, isolated_paths,
                                                   pipeline_yaml, base_cli_args):
        session = install_yolo()
        infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=base_cli_args)

        kwargs = session.instances[0].calls[0][1]
        assert kwargs["conf"] == pytest.approx(0.25)
        assert kwargs["verbose"] is False
        assert "batch" not in kwargs
        assert "workers" not in kwargs
        assert "data" not in kwargs
        assert "project" not in kwargs

    def test_zero_detections(self, install_yolo, isolated_paths, pipeline_yaml,
                             base_cli_args):
        install_yolo(n_boxes=0)
        result = infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=base_cli_args)
        assert result.success is True
        assert result.stats["detections"] == 0
        assert result.stats["per_class"] == {}

    def test_beautify_disabled_falls_back_to_plot(self, install_yolo, isolated_paths,
                                                  pipeline_yaml, base_cli_args):
        install_yolo()
        result = infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=base_cli_args,
                            beautify=False)
        assert result.success is True
        audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
        assert audit["result_summary"]["beautified"] is False
        assert len(list(result.output_dir.glob("*.jpg"))) == 4

    def test_output_dir_increments_on_duplicate_name(self, install_yolo, isolated_paths,
                                                    pipeline_yaml, base_cli_args):
        install_yolo()
        first = infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=base_cli_args)
        second = infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=base_cli_args)
        assert first.output_dir.name == "unit"
        assert second.output_dir.name == "unit2"


# ============================================================================
# 接缝 1: OutputSink
# ============================================================================
class TestOutputSinkSeam:
    def test_custom_sink_used_and_local_files_absent(self, install_yolo, isolated_paths,
                                                     pipeline_yaml, base_cli_args):
        install_yolo()
        sink = RecordingSink()
        result = infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=base_cli_args,
                            output_sink=sink)
        assert result.success is True
        assert (sink.opened, sink.frames, sink.closed) == (1, 4, 1)
        assert list(result.output_dir.glob("*.jpg")) == []

    def test_null_sink_is_default_when_save_false(self, install_yolo, isolated_paths,
                                                  pipeline_yaml, base_cli_args):
        install_yolo()
        args = dict(base_cli_args, save=False)
        result = infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=args)
        assert result.success is True
        assert result.saved is False
        assert list(result.output_dir.glob("*.jpg")) == []


# ============================================================================
# 接缝 2: Hooks
# ============================================================================
class TestHooksSeam:
    def test_all_hooks_fire(self, install_yolo, isolated_paths, pipeline_yaml,
                            base_cli_args):
        install_yolo()
        got = {"frame": 0, "progress": 0, "complete": None, "error": 0}
        hooks = InferHooks(
            on_frame=lambda e: got.__setitem__("frame", got["frame"] + 1),
            on_progress=lambda e: got.__setitem__("progress", got["progress"] + 1),
            on_complete=lambda r: got.__setitem__("complete", r),
            on_error=lambda e: got.__setitem__("error", got["error"] + 1),
            progress_interval_frames=2,
        )
        result = infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=base_cli_args,
                            hooks=hooks)
        assert got["frame"] == 4
        assert got["progress"] == 2
        assert got["error"] == 0
        assert got["complete"] is result

    def test_hook_exception_does_not_break_run(self, install_yolo, isolated_paths,
                                               pipeline_yaml, base_cli_args):
        install_yolo()

        def boom(_):
            raise RuntimeError("client callback boom")

        result = infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=base_cli_args,
                            hooks=InferHooks(on_frame=boom, on_complete=boom))
        assert result.success is True
        assert result.stats["frames"] == 4


# ============================================================================
# 接缝 3: CancelToken
# ============================================================================
class TestCancelSeam:
    def test_pre_cancelled_token_yields_empty_run(self, install_yolo, isolated_paths,
                                                  pipeline_yaml, base_cli_args):
        install_yolo()
        token = CancelToken()
        token.cancel()
        sink = RecordingSink()
        result = infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=base_cli_args,
                            output_sink=sink, cancel_token=token)
        assert result.success is True
        assert result.stats["frames"] == 0
        assert sink.frames == 0


# ============================================================================
# 失败路径
# ============================================================================
class TestFailurePaths:
    def test_model_exception_returns_failure_and_fires_error(self, install_yolo,
                                                             isolated_paths,
                                                             pipeline_yaml,
                                                             base_cli_args):
        install_yolo(fail=True)
        errors = []
        result = infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=base_cli_args,
                            hooks=InferHooks(on_error=errors.append))
        assert result.success is False
        assert "fake model exploded" in result.error
        assert len(errors) == 1

    def test_missing_source_returns_failure(self, install_yolo, isolated_paths,
                                            pipeline_yaml):
        install_yolo()
        result = infer_yolo(pipeline_yaml=pipeline_yaml,
                            cli_args={"show": False, "save": True})
        assert result.success is False
        assert "未指定推理输入源" in result.error

    def test_nonexistent_source_returns_failure(self, install_yolo, isolated_paths,
                                                pipeline_yaml, tmp_path):
        install_yolo()
        result = infer_yolo(
            pipeline_yaml=pipeline_yaml,
            cli_args={"source": str(tmp_path / "ghost"), "show": False, "save": True},
        )
        assert result.success is False
        assert result.error


# ============================================================================
# 阶段 7: 日志归档 (复制而非重命名)
# ============================================================================
class TestLogArchival:
    def test_rename_log_true_copies_log(self, install_yolo, isolated_paths,
                                        pipeline_yaml, base_cli_args, tmp_path,
                                        monkeypatch):
        log_file = tmp_path / "live.log"
        log_file.write_text("hello", encoding="utf-8")
        monkeypatch.setattr(svc, "_find_project_log_path", lambda: log_file)
        install_yolo()

        result = infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=base_cli_args,
                            rename_log=True)
        assert result.log_path == log_file
        assert log_file.exists()                       # 原日志不动
        assert (result.output_dir / "yolo11n.log").exists()   # 副本进输出目录

    def test_rename_log_false_skips_copy(self, install_yolo, isolated_paths,
                                         pipeline_yaml, base_cli_args, tmp_path,
                                         monkeypatch):
        log_file = tmp_path / "live.log"
        log_file.write_text("hello", encoding="utf-8")
        monkeypatch.setattr(svc, "_find_project_log_path", lambda: log_file)
        install_yolo()

        result = infer_yolo(pipeline_yaml=pipeline_yaml, cli_args=base_cli_args,
                            rename_log=False)
        assert list(result.output_dir.glob("*.log")) == []


# ============================================================================
# 纯函数
# ============================================================================
class TestHelpers:
    def test_resolve_output_dir_creates_suffix(self, tmp_path):
        first = svc._resolve_output_dir(tmp_path, "pred", exist_ok=False)
        second = svc._resolve_output_dir(tmp_path, "pred", exist_ok=False)
        third = svc._resolve_output_dir(tmp_path, "pred", exist_ok=False)
        assert first.name == "pred"
        assert second.name == "pred2"
        assert third.name == "pred3"

    def test_resolve_output_dir_exist_ok_reuses(self, tmp_path):
        first = svc._resolve_output_dir(tmp_path, "pred", exist_ok=True)
        second = svc._resolve_output_dir(tmp_path, "pred", exist_ok=True)
        assert first == second

    def test_resolve_model_ref_prefers_trained(self, tmp_path, monkeypatch):
        trained = tmp_path / "trained"
        pretrained = tmp_path / "pretrained"
        trained.mkdir()
        pretrained.mkdir()
        (trained / "best.pt").write_bytes(b"x")
        (pretrained / "best.pt").write_bytes(b"y")
        monkeypatch.setattr(svc, "TRAINED_MODELS_DIR", trained)
        monkeypatch.setattr(svc, "PRETRAINED_MODELS_DIR", pretrained)
        assert svc._resolve_model_ref("best.pt") == str(trained / "best.pt")

    def test_resolve_model_ref_falls_back_to_raw(self, tmp_path, monkeypatch):
        monkeypatch.setattr(svc, "TRAINED_MODELS_DIR", tmp_path / "a")
        monkeypatch.setattr(svc, "PRETRAINED_MODELS_DIR", tmp_path / "b")
        assert svc._resolve_model_ref("yolo11n.pt") == "yolo11n.pt"

    def test_copy_log_missing_src_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(svc, "_find_project_log_path", lambda: None)
        assert svc._copy_log_to_output(tmp_path, "x") is None

    def test_copy_log_preserves_source(self, tmp_path, monkeypatch):
        src = tmp_path / "a.log"
        src.write_text("content", encoding="utf-8")
        monkeypatch.setattr(svc, "_find_project_log_path", lambda: src)
        out = tmp_path / "out"
        out.mkdir()
        target = svc._copy_log_to_output(out, "run")
        assert target == out / "run.log"
        assert target.read_text(encoding="utf-8") == "content"
        assert src.exists()            # 复制而非移动

    def test_copy_log_oserror_swallowed(self, tmp_path, monkeypatch):
        src = tmp_path / "a.log"
        src.write_text("x", encoding="utf-8")
        monkeypatch.setattr(svc, "_find_project_log_path", lambda: src)

        def boom(*a, **k):
            raise OSError("locked")

        monkeypatch.setattr(svc.shutil, "copy2", boom)
        assert svc._copy_log_to_output(tmp_path, "run") is None


class TestInferStats:
    def test_zero_frames_no_division_error(self):
        s = InferStats()
        assert s.avg_fps == 0.0
        assert s.avg_latency_ms == 0.0

    def test_to_dict_rounds_and_nests_fps(self):
        s = InferStats(frames=10, detections=20, infer_time_sec=2.0,
                       capture_fps=11.11, infer_fps=22.22, render_fps=33.33,
                       loop_fps=44.44, current_fps=55.55,
                       speed_ms={"inference": 1.5})
        d = s.to_dict()
        assert d["avg_fps"] == 5.0
        assert d["avg_latency_ms"] == 200.0
        assert d["fps"]["loop"] == 44.44
        assert d["speed_ms"] == {"inference": 1.5}

    def test_log_infer_stats_runs(self, caplog):
        import logging
        s = InferStats(frames=1, detections=2, per_class={"head": 2},
                       loop_fps=10.0, speed_ms={"preprocess": 1.0})
        with caplog.at_level(logging.INFO, logger="od_platform.inference.service"):
            svc.log_infer_stats(s)
        assert "处理帧数" in caplog.text
        assert "head" in caplog.text


class TestFrameProcessor:
    def _processor(self, beautify=True, visualizer=None):
        return svc._FrameProcessor(
            model=None, predict_kwargs={}, do_beautify=beautify,
            visualizer=visualizer, use_label_mapping=True,
            style_overrides={"box_thickness": 2}, names={0: "head", 1: "反光衣"},
        )

    def test_draw_beautify_path_caches_style(self):
        viz = BeautifyVisualizer(labels=["head"])
        proc = self._processor(visualizer=viz)
        img = np.zeros((64, 64, 3), dtype=np.uint8)

        class _Boxes:
            pass

        class _Result:
            def __init__(self):
                self.boxes = None

        proc.draw(img, _Result(), [], 0)
        assert isinstance(proc._style, DrawStyle)
        assert proc._style.box_thickness == 2

    def test_draw_plot_fallback(self):
        class _Result:
            def plot(self):
                return np.full((5, 5, 3), 9, dtype=np.uint8)

        out = self._processor(beautify=False).draw(
            np.zeros((10, 10, 3), np.uint8), _Result(), [], 0)
        assert out.shape == (5, 5, 3)
