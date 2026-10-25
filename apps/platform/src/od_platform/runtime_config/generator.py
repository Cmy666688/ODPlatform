#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : generator.py
# @Author    : 雨霓同学 (ODPlatform team)
# @Project   : ODPlatform
# @Function  : runtime_config 子系统——YAML 模板生成器
"""配置文件生成器: 从 Pydantic 配置类反射生成 YAML 模板.

★ 设计核心:
   1. 反射, 不重复——所有元数据从 BaseConfig 的 get_field_groups() /
      get_field_metadata() 来, 不在 Generator 里维护一份字段表.
   2. 安全, 双闸门——默认 overwrite=False; 真覆盖时自动备份原文件.
   3. CLI 入口——主推 odp-gen-config <name> (pyproject.toml entry-point),
                 备胎 python -m od_platform.runtime_config.generator <name>.

用法:
    # 程序调用:
    from od_platform.runtime_config.generator import ConfigGenerator
    from od_platform.runtime_config.train      import YOLOTrainConfig
    gen = ConfigGenerator()
    gen.generate(YOLOTrainConfig, "train.yaml", overwrite=False)

    # CLI:
    odp-gen-config train
    odp-gen-config val --overwrite                # 覆盖前自动备份
    odp-gen-config train --overwrite --no-backup  # 显式关掉备份
    odp-gen-config train -o /tmp/my_train.yaml
"""
from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Type, Union

from od_platform.runtime_config.base import BaseConfig

from od_platform.common.paths import runtime_config_path
from od_platform.runtime_config.registry import CONFIG_REGISTRY

logger = logging.getLogger(__name__)


# ============================================================
# ConfigGenerator
# ============================================================

class ConfigGenerator:
    """从 Pydantic 配置类反射生成 YAML 模板.

    使用阶段 2 立的 get_field_groups() / get_field_metadata() 反射元数据.
    """

    def __init__(self, indent: int = 2):
        """
        Args:
            indent: YAML 缩进空格数 (目前只用于嵌套结构, 当前简化未启用)
        """
        self.indent = indent

    # ============================================================
    # 公开接口
    # ============================================================

    def generate(
        self,
        config_class: Type[BaseConfig],
        output_path: Union[str, Path],
        *,
        overwrite: bool = False,
        backup:    bool = True,
        title:     Optional[str] = None,
    ) -> bool:
        """生成配置文件.

        Args:
            config_class: Pydantic 配置类 (BaseConfig 的子类)
            output_path:  输出文件路径
            overwrite:    是否覆盖已有文件 (默认 False, 安全)
            backup:       覆盖前是否备份原文件 (默认 True).
                          仅在 overwrite=True 且 文件存在 时生效.
                          备份命名: <name>.yaml.bak.<YYYYMMDD_HHMMSS>
            title:        配置文件标题 (默认用类名)

        Returns:
            True  — 已生成
            False — 文件已存在且 overwrite=False, 未生成 (★ 不是错误)
        """
        output_path = Path(output_path)

        # 第一道防线: 不覆盖
        if output_path.exists() and not overwrite:
            logger.info(f"配置文件已存在, 跳过生成: {output_path}")
            return False

        # 第二道防线: 覆盖前备份
        if output_path.exists() and backup:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = output_path.with_name(
                f"{output_path.name}.bak.{stamp}"
            )
            shutil.copy2(output_path, backup_path)        # copy2 保留 metadata
            logger.warning(f"覆盖前已备份原配置: {backup_path}")

        # 创建父目录 (不存在的话)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 生成 YAML 内容
        content = self._generate_yaml(config_class, title)

        # 写入文件
        output_path.write_text(content, encoding="utf-8")
        logger.info(f"配置文件已生成: {output_path}")
        return True

    # ============================================================
    # 内部方法 — YAML 内容生成
    # ============================================================

    def _generate_yaml(
        self,
        config_class: Type[BaseConfig],
        title: Optional[str] = None,
    ) -> str:
        """生成完整 YAML 内容(头部 + 字段分组 + 尾部 FAQ)"""
        lines: List[str] = []

        # 文件头部
        lines.append("#" + "=" * 78)
        lines.append(f"# {title or config_class.__name__}")
        lines.append(f"# 自动生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("#" + "=" * 78)
        lines.append("")

        # 创建默认实例(用于反射 group 信息和默认值)
        config = config_class()

        # 按分组生成字段
        groups = config.get_field_groups()
        for group_name, field_names in groups.items():
            lines.append("")
            lines.append("#" + "-" * 78)
            lines.append(f"# {group_name}")
            lines.append("#" + "-" * 78)
            lines.append("")
            for field_name in field_names:
                lines.extend(self._generate_field(config, field_name))
                lines.append("")        # 字段间空行

        # 文件尾部 — 常见问题 FAQ
        lines.append("")
        lines.append("#" + "=" * 78)
        lines.append("# 常见问题")
        lines.append("#" + "=" * 78)
        lines.append("#")
        lines.append("# Q: 如何修改配置?")
        lines.append("# A: 直接编辑对应参数的值, 保存后重新运行即可")
        lines.append("#")
        lines.append("# Q: 命令行参数会覆盖配置文件吗?")
        lines.append("# A: 是的, 命令行参数优先级最高: CLI > YAML > DEFAULT")
        lines.append("#")
        lines.append("# Q: 如何恢复默认配置?")
        lines.append("# A: 删除此文件, 程序会自动报错并提示重新生成")
        lines.append("#")
        lines.append("# Q: 如何用新版默认值重新生成此模板?")
        lines.append("# A: 跑 'odp-gen-config <name> --overwrite' (会自动备份原文件)")
        lines.append("#    备胎(装包前可用): python -m od_platform.runtime_config.generator <name> --overwrite")
        lines.append("#")
        lines.append("#" + "=" * 78)

        return "\n".join(lines)

    def _generate_field(
        self,
        config: BaseConfig,
        field_name: str,
    ) -> List[str]:
        """生成单个字段的 YAML 内容(含注释 + 值)"""
        lines: List[str] = []

        # 取字段元数据
        metadata = config.get_field_metadata(field_name)

        # 主注释 (yaml_comment, fallback 到 description)
        yaml_comment = metadata.get("yaml_comment") or metadata.get("description")
        if yaml_comment:
            lines.append(f"# {yaml_comment}")

        # 示例 (最多 5 个, 太多反而花眼)
        examples = metadata.get("examples", [])
        if examples:
            examples_str = ", ".join(str(e) for e in examples[:5])
            lines.append(f"# 示例: {examples_str}")

        # 提示 (一行一条)
        tips = metadata.get("tips", [])
        if tips:
            lines.append("# 提示:")
            for tip in tips:
                lines.append(f"#   - {tip}")

        # 字段值(默认值)
        value = getattr(config, field_name)
        yaml_value = self._format_value(value)
        lines.append(f"{field_name}: {yaml_value}")

        return lines

    # ============================================================
    # 内部方法 — YAML 值格式化
    # ============================================================

    def _format_value(self, value: Any) -> str:
        """格式化 Python 值为 YAML 字符串.

        ★ 顺序很重要: bool 必须在 int 之前判(bool 是 int 子类).
        """
        if value is None:
            return "null"

        if isinstance(value, bool):       # ★ 必须在 int / 数字判断之前
            return "true" if value else "false"

        if isinstance(value, str):
            # 含 YAML 特殊字符时加引号
            if any(c in value for c in [":", "#", "[", "]", "{", "}"]):
                return f'"{value}"'
            return value

        if isinstance(value, (list, tuple)):
            if not value:
                return "[]"
            # 递归格式化元素: bool → true/false, 含特殊字符的 str → 加引号
            items = ", ".join(self._format_value(v) for v in value)
            return f"[{items}]"

        if isinstance(value, dict):
            # 当前没有顶层 dict 字段, 简化处理
            return "{}"

        # 数字 / 其他类型
        return str(value)




# ============================================================
# CLI 入口 (odp-gen-config / python -m ...generator)
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        prog="odp-gen-config",
        description="生成 YOLO 运行配置 YAML 模板",
    )
    parser.add_argument(
        "name",
        choices=list(CONFIG_REGISTRY),                # ★ 合法取值也由那张共享表说了算
        help="要生成的配置名 (train / val / infer)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="输出路径 (默认: configs/runtime/<name>.yaml)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有文件 (默认不覆盖)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="覆盖时不备份原文件 (默认会备份成 <name>.yaml.bak.<时间戳>)",
    )
    args = parser.parse_args()

    # name → 配置类: 读那张共享表 (registry.py), 不在这里另立 if-elif
    config_class, title = CONFIG_REGISTRY[args.name]

    # 输出路径: 走 paths.py SSoT
    output_path = args.output or runtime_config_path(args.name)

    # 生成
    gen = ConfigGenerator()
    success = gen.generate(
        config_class,
        output_path,
        overwrite=args.overwrite,
        backup=not args.no_backup,
        title=title,
    )

    if success:
        print(f"✓ 已生成: {output_path}")
    else:
        print(
            f"- 文件已存在, 未覆盖 (避免覆盖你已编辑的配置).\n"
            f"  路径: {output_path}\n"
            f"  如需重新生成, 加 --overwrite (覆盖前会自动备份)"
        )


if __name__ == "__main__":
    main()
