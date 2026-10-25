#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : style.py
# @Project   : ODPlatform / visualization
# @Function  : 绘制样式 (DrawStyle) —— 按图像尺寸自适应, 支持 yaml 覆盖
"""美化绘制的样式参数.

DrawStyle 是"怎么画"的唯一真相:
  - 框线粗细 / 标签字号 / 标签字色 / 标签背景不透明度
  - from_image_size(): 按图像尺寸给一套自适应默认值, 再叠加调用方覆盖

★ 颜色统一 BGR tuple: yaml 里写的是 [B, G, R] list, 由 _to_bgr_tuple 转成 tuple.
★ 覆盖字段白名单: 只认 dataclass 里真实存在的字段, 拼错的键立刻 TypeError,
  避免"yaml 写了 box_thicknes 却静默不生效"这类难查问题.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Color = tuple[int, int, int]


def to_bgr_tuple(value: Any) -> Color:
    """[B, G, R] / (B, G, R) → (B, G, R). 其它形态直接报错, 不猜."""
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    raise ValueError(f"颜色必须是 3 元素的 [B, G, R], 收到: {value!r}")


@dataclass(frozen=True)
class DrawStyle:
    """一帧绘制用的样式 (不可变值对象)."""

    box_thickness: int = 2
    text_scale: float = 0.6
    text_thickness: int = 1
    text_color: Color = (255, 255, 255)
    text_bg_alpha: float = 0.6
    label_padding: int = 4

    @classmethod
    def from_image_size(cls, height: int, width: int, **overrides: Any) -> "DrawStyle":
        """按图像尺寸生成自适应样式, 再叠加 overrides.

        Args:
            height / width: 图像尺寸 (像素), 必须为正.
            **overrides:    仅允许 DrawStyle 的字段名 (box_thickness / text_scale /
                            text_thickness / text_color / text_bg_alpha / label_padding).

        Raises:
            ValueError: 尺寸非正 / 数值越界.
            TypeError:  出现 DrawStyle 不认识的字段名.
        """
        if height <= 0 or width <= 0:
            raise ValueError(f"图像尺寸必须为正, 收到: height={height}, width={width}")

        unknown = set(overrides) - set(cls.__dataclass_fields__)
        if unknown:
            raise TypeError(
                f"DrawStyle 不认识这些字段: {sorted(unknown)}; "
                f"可用字段: {sorted(cls.__dataclass_fields__)}"
            )

        base = min(height, width)
        defaults = cls()
        params: dict[str, Any] = {
            "box_thickness": max(1, min(6, round(base / 320))),
            "text_scale": max(0.35, min(1.2, base / 1000.0)),
            "text_thickness": max(1, round(base / 640)),
            "text_color": defaults.text_color,
            "text_bg_alpha": defaults.text_bg_alpha,
            "label_padding": defaults.label_padding,
        }
        params.update(overrides)

        thickness = int(params["box_thickness"])
        if thickness < 1:
            raise ValueError(f"box_thickness 必须 ≥ 1, 收到: {params['box_thickness']!r}")

        scale = float(params["text_scale"])
        if scale <= 0:
            raise ValueError(f"text_scale 必须 > 0, 收到: {params['text_scale']!r}")

        text_thickness = int(params["text_thickness"])
        if text_thickness < 1:
            raise ValueError(f"text_thickness 必须 ≥ 1, 收到: {params['text_thickness']!r}")

        alpha = float(params["text_bg_alpha"])
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"text_bg_alpha 必须在 [0, 1], 收到: {params['text_bg_alpha']!r}")

        padding = int(params["label_padding"])
        if padding < 0:
            raise ValueError(f"label_padding 必须 ≥ 0, 收到: {params['label_padding']!r}")

        return cls(
            box_thickness=thickness,
            text_scale=scale,
            text_thickness=text_thickness,
            text_color=to_bgr_tuple(params["text_color"]),
            text_bg_alpha=alpha,
            label_padding=padding,
        )

    def to_dict(self) -> dict[str, Any]:
        """审计用的纯数据快照."""
        return {
            "box_thickness": self.box_thickness,
            "text_scale": self.text_scale,
            "text_thickness": self.text_thickness,
            "text_color": list(self.text_color),
            "text_bg_alpha": self.text_bg_alpha,
            "label_padding": self.label_padding,
        }
