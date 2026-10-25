"""掩膜可视化保持原图、概率语义与 GT 误差颜色。"""
import numpy as np
import pytest

from od_platform.visualization import MaskStyle, MaskVisualizer
from od_platform.backends.mixercseg.metrics import BinaryMetrics


def test_empty_mask_does_not_change_source():
    image = np.full((8, 10, 3), 40, dtype=np.uint8)
    out = MaskVisualizer().draw(image, np.zeros((8, 10), dtype=np.float32))
    assert np.array_equal(out, image) and out is not image


def test_overlay_only_changes_foreground_and_keeps_input():
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    probabilities = np.zeros((8, 10), dtype=np.float32); probabilities[2:5, 3:7] = 0.8
    viz = MaskVisualizer(MaskStyle(alpha=1, contour_thickness=0))
    result = viz.draw(image, probabilities)
    assert np.array_equal(result[3, 4], [0, 255, 0])
    assert not image.any() and not result[0, 0].any()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 2.0])
def test_logits_and_nonfinite_probabilities_are_rejected(bad):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        MaskVisualizer().draw(image, np.full((8, 8), bad, dtype=np.float32))


def test_comparison_shows_false_positive_and_false_negative():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    prediction = np.zeros((8, 8), dtype=np.float32); prediction[1, 1] = 1
    truth = np.zeros((8, 8), dtype=np.uint8); truth[2, 2] = 255
    result = MaskVisualizer().comparison(image, prediction, truth)
    assert result.shape == (36, 32, 3)
    assert np.array_equal(result[29, 25], [0, 0, 255])
    assert np.array_equal(result[30, 26], [255, 0, 0])


def test_binary_metrics_known_confusion_matrix():
    metrics = BinaryMetrics()
    metrics.update(np.array([0.9, 0.9, 0.1, 0.1]), np.array([1, 0, 1, 0]))
    assert metrics.to_dict()["F1"] == 0.5
    assert metrics.to_dict()["mIoU"] == pytest.approx(1 / 3)
