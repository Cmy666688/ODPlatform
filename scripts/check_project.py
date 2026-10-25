#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : check_project.py
# @Project   : ODPlatform
# @Function  : 本地 / pre-commit / CI 共用的只读检查入口
"""检查当前源码,在临时 workspace 生成演示数据并跑完整回归。

测试不借用真实 data/、runs/、模型权重或机器专用的数据集 yaml。
临时副本包含当前未提交源码,pre-commit 和手工执行采用同一检查口径。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


def _run(args: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"检查: {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, *args], cwd=cwd, env=env, check=True)


def _copy_test_workspace(target: Path) -> None:
    """只拷源码、测试、运行模板和 marker,不带入任何运行产物。"""
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")
    for name in (".odp-workspace", "pyproject.toml"):
        shutil.copy2(ROOT_DIR / name, target / name)
    for name in ("apps/platform/src", "apps/platform/tests", "apps/platform/configs/runtime",
                 "apps/desktop", "apps/web-backend/src", "apps/streamlit", "scripts", "tests"):
        source = ROOT_DIR / name
        if source.exists():
            shutil.copytree(source, target / name, ignore=ignore)


def main() -> int:
    env = dict(os.environ, PYTHONUTF8="1", QT_QPA_PLATFORM="offscreen")
    try:
        _run(["-m", "ruff", "check", "."], cwd=ROOT_DIR, env=env)
        _run(["-m", "mypy", "apps/platform/src", "apps/desktop", "apps/web-backend/src", "apps/streamlit", "scripts/check_project.py"],
             cwd=ROOT_DIR, env=env)
        with tempfile.TemporaryDirectory(prefix="odp-check-") as temp:
            workspace = Path(temp)
            _copy_test_workspace(workspace)
            # 子进程优先导入临时副本,避免 editable 安装指回真实 workspace。
            env["PYTHONPATH"] = os.pathsep.join([str(workspace / "apps/platform/src"),
                                               str(workspace / "apps/web-backend/src"),
                                               str(workspace)])
            _run(["scripts/demo_data/make_demo.py"], cwd=workspace, env=env)
            _run(["scripts/demo_data/make_broken_demo.py"], cwd=workspace, env=env)
            _run(["-m", "pytest", "-q"], cwd=workspace, env=env)
    except subprocess.CalledProcessError as e:
        return e.returncode
    print("全部检查通过。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
