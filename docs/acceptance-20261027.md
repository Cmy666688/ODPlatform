# 2026-10-27 功能与训练验收

## 正式 WSODD 实验

- 任务：`c6e42c0be10949318a22cc23d3bcbf1a`，状态 `completed`。
- NVIDIA RTX 4090 24GB，Torch 2.10.0+cu128、Ultralytics 8.4.183；50/50 epochs。
- 训练现场：`runs/training/20261026-220150/`；评估：`runs/web_jobs/<任务 ID>/evaluation/`。
- 原数据 7467 张、14 类，训练/验证/测试为 5638/1052/777。测试不参与选模。
- 最佳与最后权重、训练曲线、逐类 CSV、PR/混淆矩阵、测试报告和 **48/48 张对比图**均已回传。同步时间为 2026-10-27 07:59:14（北京时间）。

| 独立测试指标 | 结果 |
| --- | ---: |
| mAP50 | 16.92% |
| mAP50–95 | 8.18% |
| 宏平均 precision | 47.93% |
| 宏平均 recall | 21.82% |
| 上述宏平均 P/R 的调和均值 F1 | 29.99% |

该轮已经完成训练和报告生成，但精度偏低，验收标记保留 `needs_review`。验证集缺 tree、测试集缺 animal；缺失类别不虚构 AP。该记录不能支持“业务效果达标”的结论。

## 视频与解释链路

- 任务：`6fc3ab9a7b2b4d2e896c20e58eb73e49`，状态 `completed`。
- 输入：`runs/wsodd_acceptance/video/wsodd_static_acceptance.mp4`；来源说明：同目录 `acceptance.json`。
- 输入由一张 WSODD 测试静态图重复生成：960×540、6 FPS、12 帧、2 秒。不是实际连续拍摄视频。
- 使用本次正式训练权重，在远程 RTX 4090 CUDA 执行真实 YOLO 推理，本机不执行模型计算。
- 输出：`runs/detection_infer/20261027-080319/annotated.mp4`（H.264）、`frames.jsonl`、`video_summary.json`、抽样检测框与 SHA256 审计。
- 12/12 帧处理成功；boat 出现在 11 帧，同帧最多 1 个，最高置信度约 0.28737。帧数不等于独立目标数。
- 本机将结构化摘要交给真实 DeepSeek API，`runs/web_jobs/<视频任务 ID>/explanation.json` 状态 `completed`。原视频、模型权重、SSH 配置与密钥均不作为解释上下文发送。
- 浏览器已验证 H.264 视频加载、结果摘要与真实解释展示；HTTP Range / 非法 Range / 下载通过接口测试。

此测试验收上传、远程执行、编码、同步、网页与大模型串联。不能用于跟踪精度、真实视频泛化、航速或碰撞风险验收。

## 其他收尾

- 统一 D3/D4 CLI 进度与当前检测质检；图像完整解码、框/类别范围、SHA 身份及历史对比。
- runs 保留策略、预览、引用/活动进程保护、隔离回收与恢复。默认关闭，真实数据未删除。
- 项目介绍、数据理解、运行维护、视频分析页及 Streamlit 多数据集筛选。
- Gradio D7 演示复用 Django 任务和模型；单独刷新结果不会重复提交推理。
- DeepSeek 页面只显示配置状态，无密钥明文或密钥配置示例。
- `scripts/check_project.py` 作为 pre-commit / CI 入口，在隔离工作区运行 Ruff、Mypy 与完整 pytest，不借用真实数据或权重。

## 启动入口

```powershell
pip install -e "./apps/platform[dev,desktop]"
pip install -e "./apps/web-backend[workbench,test,demo]"
powershell -ExecutionPolicy Bypass -File scripts/start_web.ps1 -Workbench -Demo
```

Django：<http://127.0.0.1:8000/#project>；Streamlit：<http://127.0.0.1:8501>；Gradio：<http://127.0.0.1:7860>。
视频推理的远程 Python 需要 `imageio-ffmpeg>=0.6,<1`，本次已安装；GPU 缺失时仍可浏览历史产物。
