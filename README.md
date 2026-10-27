# ODPlatform

通用目标检测开发平台（Monorepo）。

当前默认 **WSODD / YOLOv8n 港区水域多类别目标检测**。提供 VOC/YOLO 大包上传、
数据质检、远程 GPU 训练、独立测试、图片/视频可视化与 DeepSeek 摘要解释。
本轮 50 轮训练及产物回传已完成，精度仍待改善；见 [验收记录](docs/acceptance-20261027.md)
和 [工作清单](docs/task-completion.md)。

历史 MixerCSeg 裂缝语义分割 CLI 保留，使用说明见 [MixerCSeg 文档](docs/mixercseg.md)：

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
pip install -e "./apps/web-backend[workbench,test,demo]"
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
pip install -e ./apps/platform
pip install -e "./apps/web-backend[workbench,test,demo]"
powershell -ExecutionPolicy Bypass -File scripts/start_web.ps1 -Workbench -Demo
```

面板打开 <http://127.0.0.1:8000/#overview>。WSODD 共 7,467 张图像、14 类，数据与配方见 [WSODD 实验说明](docs/wsodd-training.md)。训练、独立测试及图片/视频推理统一使用远程 CUDA GPU；连接配置见 [Web 说明](apps/web-backend/README.md)。新网页导入使用多类别检测标签；刷新和保存配置不会启动训练。
项目介绍、数据理解、视频分析和运行维护均可从侧栏进入。Streamlit 工作台为 <http://127.0.0.1:8501>，Gradio D7 演示为 <http://127.0.0.1:7860>。
运行保留策略默认关闭，可在维护页预览、启用和恢复；先隔离回收，延迟至少 7 天才清理，保护活动和被任务引用的目录。
功能、任务接口和验收边界见 [Web 使用说明](apps/web-backend/README.md)。
