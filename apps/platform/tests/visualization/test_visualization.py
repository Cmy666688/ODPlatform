"""visualization 包单元测试: 样式 / 检测值对象 / 美化绘制."""
from __future__ import annotations

import numpy as np
import pytest

from od_platform.visualization import (
    BeautifyVisualizer,
    Detection,
    DrawStyle,
    to_bgr_tuple,
)


# ============================================================================
# to_bgr_tuple
# ============================================================================
class TestToBgrTuple:
    def test_list_to_tuple(self):
        assert to_bgr_tuple([1, 2, 3]) == (1, 2, 3)

    def test_tuple_passthrough(self):
        assert to_bgr_tuple((4, 5, 6)) == (4, 5, 6)

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="3 元素"):
            to_bgr_tuple([1, 2])

    def test_wrong_type_raises(self):
        with pytest.raises(ValueError):
            to_bgr_tuple("green")


# ============================================================================
# DrawStyle
# ============================================================================
class TestDrawStyle:
    def test_adaptive_scales_with_size(self):
        small = DrawStyle.from_image_size(240, 320)
        large = DrawStyle.from_image_size(2160, 3840)
        assert small.box_thickness < large.box_thickness
        assert small.text_scale < large.text_scale
        assert small.box_thickness >= 1
        assert large.text_scale <= 1.2

    def test_overrides_applied(self):
        s = DrawStyle.from_image_size(100, 100, box_thickness=5, text_bg_alpha=0.3,
                                      text_color=[1, 2, 3])
        assert s.box_thickness == 5
        assert s.text_bg_alpha == 0.3
        assert s.text_color == (1, 2, 3)

    def test_unknown_field_raises_type_error(self):
        with pytest.raises(TypeError, match="不认识"):
            DrawStyle.from_image_size(100, 100, box_thicknes=3)

    @pytest.mark.parametrize("kwargs", [
        {"box_thickness": 0},
        {"text_scale": 0},
        {"text_thickness": 0},
        {"text_bg_alpha": 1.5},
        {"label_padding": -1},
        {"text_color": [1, 2]},
    ])
    def test_invalid_override_raises(self, kwargs):
        with pytest.raises(ValueError):
            DrawStyle.from_image_size(100, 100, **kwargs)

    def test_zero_size_raises(self):
        with pytest.raises(ValueError, match="尺寸"):
            DrawStyle.from_image_size(0, 100)

    def test_to_dict_is_json_safe(self):
        d = DrawStyle.from_image_size(480, 640).to_dict()
        assert isinstance(d["text_color"], list)
        assert isinstance(d["box_thickness"], int)


# ============================================================================
# Detection
# ============================================================================
class TestDetection:
    def test_xyxy_must_be_4(self):
        with pytest.raises(ValueError):
            Detection(xyxy=(1, 2, 3))

    def test_confidence_optional(self):
        assert Detection(xyxy=(0, 0, 1, 1)).confidence is None


# ============================================================================
# BeautifyVisualizer
# ============================================================================
class TestBeautifyVisualizer:
    def _viz(self, **kwargs):
        base = {
            "labels": ["head", "反光衣"],
            "label_mapping": {"head": "头部"},
            "color_mapping": {"head": [0, 255, 0]},
            "default_color": (0, 0, 255),
        }
        base.update(kwargs)
        return BeautifyVisualizer(**base)

    def test_from_yolo_results_converts(self):
        dets = BeautifyVisualizer.from_yolo_results(
            boxes=np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
            confidences=np.array([0.8, 0.6]),
            labels=["a", "b"],
        )
        assert len(dets) == 2
        assert dets[0].xyxy == (1.0, 2.0, 3.0, 4.0)
        assert dets[1].confidence == pytest.approx(0.6)

    def test_from_yolo_results_empty(self):
        dets = BeautifyVisualizer.from_yolo_results(None, None, [])
        assert dets == []

    def test_from_yolo_results_short_labels(self):
        dets = BeautifyVisualizer.from_yolo_results(
            [[1, 1, 2, 2]], [0.5], [])
        assert dets[0].label == ""

    def test_draw_returns_copy_and_does_not_mutate_input(self):
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        original = img.copy()
        out = self._viz().draw(img, [Detection((5, 5, 50, 50), "head", 0.9)])
        assert np.array_equal(img, original)
        assert not np.array_equal(out, original)

    def test_draw_colors_known_label(self):
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        out = self._viz().draw(
            img, [Detection((10, 10, 60, 60), "head")],
            style=DrawStyle(box_thickness=1, text_bg_alpha=0.0),
        )
        assert tuple(out[55, 10]) == (0, 255, 0)

    def test_draw_uses_default_color_for_unknown_label(self):
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        out = self._viz().draw(
            img, [Detection((10, 10, 60, 60), "反光衣")],
            style=DrawStyle(box_thickness=1),
        )
        assert tuple(out[55, 10]) == (0, 0, 255)

    def test_draw_skips_degenerate_box(self):
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        out = self._viz().draw(img, [Detection((30, 30, -5, -5), "head")])
        assert np.array_equal(out, img)

    def test_draw_rejects_bad_image(self):
        with pytest.raises(ValueError):
            self._viz().draw(np.zeros((10, 10), dtype=np.uint8), [])

    def test_display_text_mapping_switch(self):
        viz = self._viz()
        assert viz.display_text("head", 0.88, True) == "头部 0.88"
        assert viz.display_text("head", None, False) == "head"

    def test_missing_font_falls_back(self, caplog):
        viz = self._viz(font_path="C:/definitely/not/here.ttf")
        assert viz.font_path is None
        assert "字体文件不存在" in caplog.text

    def test_draw_survives_single_box_exception(self, monkeypatch):
        """单框绘制异常只跳过该框, 不炸整帧 (永不抛纪律)."""
        img = np.zeros((60, 60, 3), dtype=np.uint8)
        viz = self._viz()
        dets = [Detection((5, 5, 20, 20), "head"), Detection((10, 10, 40, 40), "head")]
        calls = {"n": 0}

        def bad_display(label, conf, mapping):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return "ok"

        monkeypatch.setattr(viz, "display_text", bad_display)
        out = viz.draw(img, dets)
        assert out.shape == img.shape
        assert calls["n"] == 2

    def test_unknown_mapping_key_warns(self, caplog):
        self._viz(label_mapping={"person": "人员"})
        assert "模型不存在的类别" in caplog.text
