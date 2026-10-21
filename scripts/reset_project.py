#!/usr/bin/env python
# @FileName  : scripts/reset_project.py
# @Author    : ODPlatform Team
# @Function  : 项目重置的【开发阶段入口】(无需安装 package)
#
# 用法:
#   python scripts/reset_project.py [flags]   # flags 与 odp-reset 一致
#
# 注意:必须调 main() 而不是 reset_project()——argparse 解析在 main() 里,
# 直接调 reset_project() 会让所有 flag 默默拿默认值。
"""ODPlatform 项目重置入口(开发阶段)"""

import sys
from pathlib import Path

# 从仓库根定位 platform 的 src 目录
REPO_ROOT = Path(__file__).resolve().parent.parent
PLATFORM_SRC = REPO_ROOT / "apps" / "platform" / "src"

# 把 src 加到 sys.path 最前面(优先于已安装版本)
sys.path.insert(0, str(PLATFORM_SRC))

# 导入真正的入口——调 main(负责 argparse),不是 reset_project(业务函数)
from od_platform.cli.reset_project import main

if __name__ == "__main__":
    sys.exit(main())
