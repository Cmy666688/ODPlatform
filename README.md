# ODPlatform

通用目标检测开发平台（Monorepo）。

已接入 **MixerCSeg 裂缝语义分割**，默认示例为较小的 CrackMap 数据集。
数据质检、原始训练参数、HOG 权重差异和 D7 掩膜可视化见
[MixerCSeg 使用说明](docs/mixercseg.md)。原有 YOLO 命令保持兼容。

```bash
pip install -e "./apps/platform[dev,desktop,mixercseg]"
odp-prepare-cracks --dataset CrackMap
odp-train --backend mixercseg --config mixercseg_train
# 官方权重下载、推理与可视化命令见 docs/mixercseg.md
```

- `apps/platform/` — 核心引擎（数据转换、训练、推理）
- `apps/web-backend/` — Django 服务、Vue 数据集管理与训练监控面板
- `apps/streamlit/` — 复用 API 的算法调试工作台
- `apps/desktop/` — PySide6 桌面推理示例（可选）
- `data/` — 数据集（共享，gitignored）
- `models/` — 模型权重（共享，gitignored）
- `runs/` — 训练运行现场（gitignored）

快速开始：

```bash
pip install -e ./apps/platform
odp-init
```

详见 `docs/architecture/`。

开发与检查：

```bash
pip install -e "./apps/platform[dev,desktop]"
pip install -e "./apps/web-backend[workbench,test]"
python scripts/check_project.py
python -m pre_commit install
```

`check_project.py` 依次运行 Ruff、Mypy 和完整 pytest。测试在临时 workspace
生成演示数据，不读取或覆盖当前项目的数据集、权重、运行记录。Git 提交前和
GitHub Actions 共用这一个入口；检查失败会拦停。Ruff 保留现有分段导入、
`typing.Dict/Optional` 和紧凑写法，不自动格式化或修改源码。

桌面推理：

```bash
python -m apps.desktop.main
```

桌面用法及线程收尾说明见 [apps/desktop/README.md](apps/desktop/README.md)。
开发检查与 CI 说明见 [docs/development.md](docs/development.md)。

Web 监控与结果可视化：

```powershell
pip install -e "./apps/platform[mixercseg,cpu]"
pip install -e "./apps/web-backend[workbench,test]"
powershell -ExecutionPolicy Bypass -File scripts/start_web.ps1 -Workbench
```

面板打开 <http://127.0.0.1:8000/#overview>。当前默认使用 **WSODD / YOLOv8n 水面目标检测**，7,467 张图像、14 类，支持检测框预览、远程训练和独立测试可视化；数据与配方见 [WSODD 实验说明](docs/wsodd-training.md)。网页训练、独立测试及图片推理统一使用远程 CUDA GPU；连接配置见 [Web 说明](apps/web-backend/README.md)。原二值分割 ZIP / 文件夹导入作为兼容入口保留，刷新和保存配置不会启动训练。
功能、任务接口和验收边界见 [Web 使用说明](apps/web-backend/README.md)。
