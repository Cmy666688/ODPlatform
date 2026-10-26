# 开发检查与 CI

## 环境

核心引擎支持 Python 3.10 及以上；包含 Web/Streamlit 的完整开发环境使用 Python 3.11 及以上。
在仓库根安装开发工具与可选端依赖：

```bash
pip install -e "./apps/platform[dev,desktop,mixercseg,cpu]"
pip install -e "./apps/web-backend[workbench,test]"
```

Windows 下建议设置 `PYTHONUTF8=1`，避免重定向日志时中文编码受终端影响。

## 检查入口

```bash
python scripts/check_project.py
```

检查顺序：

1. `python -m ruff check .`：检查源码中的名称、语法和易错写法。
2. Mypy：检查引擎、桌面、Web 后端、Streamlit 和检查入口的类型。
3. 在临时 workspace 中生成 demo / broken 数据并运行完整 pytest。

临时副本包含当前工作区的源码、测试、运行模板和 `.odp-workspace`，不包含真实
`data/`、`models/`、`runs/` 及数据集 yaml。测试通过 `PYTHONPATH` 优先导入副本，
editable 安装不会把测试重新导向真实项目。结束后自动回收临时目录。

Ruff 不启用会迁移旧写法的 `UP` / `I` 规则，并允许原有同行短句；保留其他
错误检查。本地和 CI 使用相同 Ruff 版本，不挂自动格式化或 `--fix`。
Mypy 跳过 torch / ultralytics 等外部包内部分析，项目源码仍完整检查。

## 提交前检查

```bash
python -m pre_commit install
python -m pre_commit run --all-files
```

hook 调用同一个检查入口。请在已安装项目依赖的 Python 环境中提交；检查失败
会阻止提交，可按输出定位修复。正常提交时 pre-commit 会暂存工作区中未暂存的
改动，只检查本次准备提交的版本；`--all-files` 用于手工检查当前完整工作区。

## GitHub Actions

`.github/workflows/ci.yml` 在 push、pull_request 和手动触发时运行：

- Ubuntu / Python 3.11、Windows / Python 3.13。
- 安装引擎、CPU 扫描、Web 和 Streamlit 依赖，使用 offscreen Qt 验证桌面接缝。
- 执行同一个 `check_project.py`，随后构建引擎和 Web 端的源码包与 wheel。

工作流只读仓库，不发布包。需要将检查设为 PR 合并条件时，在仓库的分支保护
设置中把对应 check 设为 required。
