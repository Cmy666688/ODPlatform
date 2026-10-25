#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : visualizer.py
# @Project   : ODPlatform / visualization
# @Function  : 美化绘制器 (BeautifyVisualizer) —— 按类别配色 + 中英标签映射
"""推理结果美化绘制.

职责只有一件: 把 Detection 列表画到 BGR 图上, 比 YOLO 原生 result.plot() 好看.

能力:
  - 类别 → 颜色映射 (BGR), 没配的用 default_color
  - 英文标签 → 中文标签映射 (use_label_mapping=False 时全显英文)
  - 标签带置信度 (confidence 为 None 时不显示)
  - 中文标签: 传 font_path 走 PIL 渲染; 不传走 cv2.putText (中文会乱码, 仅适合英文)

★ 框架无关: 只吃 Detection, 不吃 ultralytics 的 Results.
  从 YOLO Results 提取字段的转换放在 from_yolo_results().
★ 永不抛: draw() 单框异常只跳过该框, 不影响整帧输出.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .detection import Detection
from .style import Color, DrawStyle, to_bgr_tuple

logger = logging.getLogger(__name__)

_FONT_BASE_SIZE = 22          # text_scale=0.6 对应的 PIL 字号
_FONT_BASE_SCALE = 0.6


def _clip_box(xyxy: Sequence[float], height: int, width: int) -> tuple[int, int, int, int]:
    """把框裁进画面并取整. 返回 (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = (int(round(float(v))) for v in xyxy)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


class BeautifyVisualizer:
    """按类别配色 + 标签映射的美化绘制器."""

    def __init__(
        self,
        labels: Optional[Sequence[str]] = None,
        label_mapping: Optional[dict[str, str]] = None,
        color_mapping: Optional[dict[str, Any]] = None,
        default_color: Any = (0, 255, 0),
        font_path: Optional[str] = None,
    ) -> None:
        """
        Args:
            labels:        模型类别名列表 (只用于自检/日志).
            label_mapping: 英文 → 显示名 映射 (如 {"person": "人员"}).
            color_mapping: 类别名 → BGR 颜色 (list 或 tuple).
            default_color: 未配置类别的颜色 (BGR).
            font_path:     中文字体路径; None/不存在 → 退回 cv2.putText.
        """
        self.labels: list[str] = list(labels or [])
        self.label_mapping: dict[str, str] = dict(label_mapping or {})
        self.color_mapping: dict[str, Color] = {
            str(k): to_bgr_tuple(v) for k, v in (color_mapping or {}).items()
        }
        self.default_color: Color = to_bgr_tuple(default_color)
        self.font_path: Optional[str] = None
        self._fonts: dict[int, Any] = {}

        if font_path:
            if Path(font_path).exists():
                self.font_path = str(font_path)
            else:
                logger.warning("字体文件不存在, 中文标签退回 cv2 渲染 (可能乱码): %s", font_path)

        # 自检: 映射里出现模型没有的类别, 只提示不报错 (配置可复用)
        unknown = set(self.label_mapping) - set(self.labels)
        if self.labels and unknown:
            logger.warning("label_mapping 中有模型不存在的类别 (忽略): %s", sorted(unknown))

    # ------------------------------------------------------------------
    # 从 ultralytics Results 提取
    # ------------------------------------------------------------------
    @staticmethod
    def from_yolo_results(
        boxes: Any,
        confidences: Any,
        labels: Sequence[str],
    ) -> list[Detection]:
        """把 ultralytics 的 boxes/confidences/labels 转成 Detection 列表.

        Args:
            boxes:       (N, 4) 的 xyxy 数组 (numpy 或可 np.asarray 的对象).
            confidences: (N,) 的置信度数组; None 表示不显示置信度.
            labels:      长度 N 的类别名列表.
        """
        box_arr = np.asarray(boxes, dtype=float).reshape(-1, 4) if boxes is not None \
            else np.zeros((0, 4), dtype=float)
        conf_arr = np.asarray(confidences, dtype=float).reshape(-1) \
            if confidences is not None else None
        label_list = list(labels or [])

        dets: list[Detection] = []
        for i, box in enumerate(box_arr):
            dets.append(Detection(
                xyxy=tuple(float(v) for v in box),          # type: ignore[arg-type]
                label=label_list[i] if i < len(label_list) else "",
                confidence=(float(conf_arr[i])
                            if conf_arr is not None and i < len(conf_arr) else None),
            ))
        return dets

    # ------------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------------
    def color_for(self, label: str) -> Color:
        return self.color_mapping.get(label, self.default_color)

    def display_text(self, label: str, confidence: Optional[float],
                     use_label_mapping: bool) -> str:
        """标签 + (可选) 置信度."""
        text = self.label_mapping.get(label, label) if use_label_mapping else label
        if confidence is not None:
            text = f"{text} {confidence:.2f}"
        return text.strip()

    def draw(
        self,
        image: np.ndarray,
        detections: Iterable[Detection],
        *,
        style: Optional[DrawStyle] = None,
        use_label_mapping: bool = True,
    ) -> np.ndarray:
        """在 image 的副本上画框 + 标签, 返回新图 (BGR)."""
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise ValueError("image 必须是 HxWx3 的 BGR ndarray")

        img = image.copy()
        style = style or DrawStyle.from_image_size(img.shape[0], img.shape[1])
        dets = list(detections or [])

        if self.font_path:
            return self._draw_with_pil(img, dets, style, use_label_mapping)
        return self._draw_with_cv2(img, dets, style, use_label_mapping)

    # ---- cv2 路径 (英文标签 / 无中文字体) ----
    def _draw_with_cv2(self, img: np.ndarray, dets: list[Detection],
                       style: DrawStyle, use_label_mapping: bool) -> np.ndarray:
        import cv2

        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        pad = style.label_padding

        for det in dets:
            try:
                color = self.color_for(det.label)
                x1, y1, x2, y2 = _clip_box(det.xyxy, h, w)
                if x2 <= x1 or y2 <= y1:
                    continue
                cv2.rectangle(img, (x1, y1), (x2, y2), color, style.box_thickness)

                text = self.display_text(det.label, det.confidence, use_label_mapping)
                if not text:
                    continue
                (tw, tht), baseline = cv2.getTextSize(
                    text, font, style.text_scale, style.text_thickness)
                box_w, box_h = tw + 2 * pad, tht + baseline + 2 * pad
                ty = y1 - box_h if y1 - box_h >= 0 else y1
                tx2 = min(w, x1 + box_w)
                ty2 = min(h, ty + box_h)

                overlay = img.copy()
                cv2.rectangle(overlay, (x1, ty), (tx2, ty2), color, -1)
                cv2.addWeighted(overlay, style.text_bg_alpha,
                                img, 1.0 - style.text_bg_alpha, 0, img)
                cv2.putText(img, text, (x1 + pad, ty + pad + tht), font,
                            style.text_scale, style.text_color,
                            style.text_thickness, cv2.LINE_AA)
            except Exception as e:      # 单框失败只跳过该框
                logger.warning("绘制单个检测框失败, 跳过: %s", e)
        return img

    # ---- PIL 路径 (中文标签) ----
    def _draw_with_pil(self, img: np.ndarray, dets: list[Detection],
                       style: DrawStyle, use_label_mapping: bool) -> np.ndarray:
        import cv2
        from PIL import Image, ImageDraw

        h, w = img.shape[:2]
        pad = style.label_padding
        font_size = max(12, int(round(_FONT_BASE_SIZE * style.text_scale / _FONT_BASE_SCALE)))
        font = self._load_font(font_size)

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        text_rgb = (style.text_color[2], style.text_color[1], style.text_color[0])

        for det in dets:
            try:
                b, g, r = self.color_for(det.label)
                x1, y1, x2, y2 = _clip_box(det.xyxy, h, w)
                if x2 <= x1 or y2 <= y1:
                    continue
                draw.rectangle([x1, y1, x2, y2], outline=(r, g, b),
                               width=style.box_thickness)

                text = self.display_text(det.label, det.confidence, use_label_mapping)
                if not text:
                    continue
                left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                tw, tht = right - left, bottom - top
                box_h = tht + 2 * pad
                ty = y1 - box_h if y1 - box_h >= 0 else y1
                draw.rectangle([x1, ty, min(w - 1, x1 + tw + 2 * pad), min(h - 1, ty + box_h)],
                               fill=(r, g, b))
                draw.text((x1 + pad - left, ty + pad - top), text,
                          font=font, fill=text_rgb)
            except Exception as e:
                logger.warning("绘制单个检测框失败, 跳过: %s", e)
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    def _load_font(self, size: int):
        from PIL import ImageFont
        assert self.font_path is not None
        if size not in self._fonts:
            self._fonts[size] = ImageFont.truetype(self.font_path, size)
        return self._fonts[size]
