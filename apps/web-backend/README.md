# apps/web-backend — Web API 与训练监控面板

状态：已实现本机 FastAPI 服务、正式监控页面和独立任务进程。

本子项目提供 ODPlatform 的 Web 后端服务,基于:

- FastAPI(REST API)
- 复用 `apps/platform/` 的核心引擎(数据转换、推理)——后端和 platform 装在同一个环境里
  (platform 已 `pip install -e`),`import od_platform` 直接能用,不涉及打包发布;
  后端 `pyproject.toml` 把 `od-platform` 写进依赖只是把这层关系记录在案
- 数据结构直接复用 platform 引擎暴露的类型

## 安装与运行

Web 开发环境使用 Python 3.11 及以上。

```powershell
pip install -e "./apps/platform[mixercseg,cpu]"
pip install -e "./apps/web-backend[workbench]"
powershell -ExecutionPolicy Bypass -File scripts/start_web.ps1 -Workbench
```

正式面板：<http://127.0.0.1:8000>。也可用 `odp-web --port 8000` 单独启动后端和面板。
入口绑定本机回环地址，静态资源随 Python 包提供，不依赖 Node 构建、外部字体或 CDN。

## 功能与边界

- 监控概览：真实 epoch、loss、验证指标、耗时、预计剩余、CPU/内存。
- GPU 可用时读取 NVML 使用率与显存；不可读取时显示暂不可用，不用 API 进程的显存代替训练进程。
- 数据质检：CrackMap 实际检查和 13 组潜在同源跨集风险。
- 实验记录：独立进程执行、JSON 持久状态、日志、停止、产物下载。
- 结果可视化：图片上传推理、阈值设置、官方参考与本次训练结果分别展示。
- 完整训练结束：自动测试 best 权重，生成指标 JSON、逐图 CSV、曲线 PNG 和对比图。

训练默认要求 CUDA，就绪前不会自动回退到 CPU 启动。配置草稿保存在浏览器本地，
保存和刷新均不会提交训练。当前完整训练仍待 GPU；既有参考图来自官方权重。

同一时间执行一个计算任务；页面关闭和后端重启不主动杀死 worker。
停止为协作式，在当前批次计算完成后检查信号；保留已完成轮次的权重，不承诺断点续训。
单进程 API（默认启动方式）负责串行受理任务，不要用多个 Uvicorn worker 启动此本机版本。
文件和状态存于 `runs/web_jobs/`；上传图像限 20 MB，使用服务端生成的文件 ID。

完整训练采用 `configs/runtime/mixercseg_full.yaml`：50 epochs、512×512、batch=1、
AdamW / PolyLR、原始 BCE/Dice。用 val 选 best、独立 test 验收。
验收目标未填写时明确显示“待人工评审”，不把运行完成等同于达到生产精度。

## 主要接口

| 接口 | 用途 |
|---|---|
| `GET /api/overview` | 环境、数据、质检与任务摘要 |
| `GET /api/config` | 原始训练配置和 JSON Schema |
| `POST /api/datasets/crackmap/validate` | 重新质检 |
| `POST /api/uploads` | 单图上传，返回文件 ID |
| `POST /api/jobs` | 提交训练或推理任务 |
| `GET /api/jobs/{id}` | 状态与指标 |
| `POST /api/jobs/{id}/cancel` | 请求停止 |
| `GET /api/jobs/{id}/artifacts` | 产物列表 |
| `GET /api/reference` | 已有官方权重参考图 |

交互式 API 文档：<http://127.0.0.1:8000/docs>。

参考: `docs/architecture/ADR-001-monorepo.md`
