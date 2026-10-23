# apps/platform/src/od_platform/data_pipeline/convert/converters/_geometry.py
"""几何换算:VOC/COCO 的绝对坐标 → YOLO 归一化。内部工具,不注册、不被自动发现。"""
from collections.abc import Sequence


def normalize_box(x1: float, y1: float, x2: float, y2: float,
                  W: float, H: float) -> tuple[float, float, float, float]:
    """(左上右下, 宽高) → (cx, cy, w, h),全部归一化到 [0,1]。"""
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def normalize_polygon(points: Sequence[float], W: float, H: float) -> list[float]:
    """COCO 分割多边形 [x1,y1,x2,y2,...] → 归一化,奇数位除 W、偶数位除 H。"""
    return [points[i] / W if i % 2 == 0 else points[i] / H for i in range(len(points))]
