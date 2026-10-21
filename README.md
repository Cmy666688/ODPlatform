# ODPlatform

通用目标检测开发平台（Monorepo）。

- `apps/platform/` — 核心引擎（数据转换、训练、推理）
- `apps/web-backend/` — Web 后端服务（占位，V1.1）
- `apps/desktop/` — 桌面端（占位）
- `data/` — 数据集（共享，gitignored）
- `models/` — 模型权重（共享，gitignored）
- `runs/` — 训练运行现场（gitignored）

快速开始：

```bash
pip install -e ./apps/platform
odp-init
```

详见 `docs/architecture/`。
