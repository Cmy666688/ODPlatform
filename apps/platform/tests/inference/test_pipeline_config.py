"""pipeline_config 单元测试: infer_pipeline.yaml 解析."""
from __future__ import annotations

from pathlib import Path

import pytest

from od_platform.inference.pipeline_config import PipelineConfig, load_pipeline_config


class TestLoadPipelineConfig:
    def test_missing_file_uses_defaults(self, tmp_path: Path, caplog):
        pc = load_pipeline_config(tmp_path / "nope.yaml")
        assert isinstance(pc, PipelineConfig)
        assert pc.viz_enabled is True
        assert pc.label_mapping == {}
        assert "不存在" in caplog.text

    def test_none_uses_default_path(self, isolated_paths, caplog):
        # isolated_paths 把默认目录指向空 tmp → 走默认值
        pc = load_pipeline_config(None)
        assert pc.viz_enabled is True
        assert "不存在" in caplog.text

    def test_valid_yaml_parsed(self, pipeline_yaml: Path):
        pc = load_pipeline_config(pipeline_yaml)
        assert pc.camera_raw == {"width": 640, "height": 480, "fps": 25}
        assert pc.viz_enabled is True
        assert pc.label_mapping == {"head": "头部"}
        assert pc.color_mapping == {"head": (0, 255, 0)}
        assert pc.default_color == (0, 0, 255)
        assert pc.style_overrides == {"box_thickness": 3}
        assert pc.font_path is None

    def test_disabled_viz(self, tmp_path: Path):
        p = tmp_path / "p.yaml"
        p.write_text("visualization:\n  enabled: false\n", encoding="utf-8")
        assert load_pipeline_config(p).viz_enabled is False

    def test_use_label_mapping_false(self, tmp_path: Path):
        p = tmp_path / "p.yaml"
        p.write_text("visualization:\n  use_label_mapping: false\n", encoding="utf-8")
        assert load_pipeline_config(p).use_label_mapping is False

    def test_invalid_yaml_type_uses_defaults(self, tmp_path: Path, caplog):
        p = tmp_path / "bad.yaml"
        p.write_text("- just\n- a\n- list\n", encoding="utf-8")
        pc = load_pipeline_config(p)
        assert pc.viz_enabled is True
        assert "顶层必须是映射" in caplog.text

    def test_invalid_color_raises(self, tmp_path: Path):
        p = tmp_path / "p.yaml"
        p.write_text("visualization:\n  color_mapping:\n    head: [1, 2]\n",
                     encoding="utf-8")
        with pytest.raises(ValueError):
            load_pipeline_config(p)


class TestCameraConfigBuilding:
    def test_build_camera_config_ok(self, pipeline_yaml: Path):
        cfg = load_pipeline_config(pipeline_yaml).build_camera_config()
        assert cfg is not None
        assert cfg.width == 640 and cfg.fps == 25

    def test_empty_camera_returns_none(self):
        assert PipelineConfig().build_camera_config() is None

    def test_invalid_camera_returns_none_with_warning(self, tmp_path: Path, caplog):
        p = tmp_path / "p.yaml"
        p.write_text("frame_source:\n  camera:\n    width: -1\n", encoding="utf-8")
        cfg = load_pipeline_config(p).build_camera_config()
        assert cfg is None
        assert "camera 配置无效" in caplog.text


class TestToAudit:
    def test_audit_shape(self, pipeline_yaml: Path):
        audit = load_pipeline_config(pipeline_yaml).to_audit()
        assert audit["viz_enabled"] is True
        assert audit["label_mapping_n"] == 1
        assert audit["color_mapping_n"] == 1
        assert audit["default_color"] == [0, 0, 255]
        assert audit["camera"] == {"width": 640, "height": 480, "fps": 25}
        assert audit["style_overrides_n"] == 1
