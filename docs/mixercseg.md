# MixerCSeg / CrackMap 接入与数据检查

本次使用 [MixerCSeg](https://github.com/spiderforest/MixerCSeg) 的 CrackMap 数据，
接入其原始模型和训练配方。选择 CrackMap 是因为上游压缩包约 18.4 MB、共 120 张，
比 DeepCrack 的约 67 MB、527 张更适合先跑通链路。原有 YOLO 链路仍可使用。

上游固定在提交 `7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d`。
模型、默认参数与来源哈希均随代码保留，不在运行时跟随 main 分支变化。

## 安装和数据准备

在仓库根执行：

```bash
pip install -e "./apps/platform[dev,desktop,mixercseg]"
odp-prepare-cracks --dataset CrackMap
```

原始 archive 留在 `data/raw/mixercseg/archives/`，原始数据单独解压到
`data/raw/mixercseg/CrackMap/`。标准化后写到 `data/processed/crackmap/`：
`images/{train,val,test}` 是 RGB 图像，`masks/{train,val,test}` 是单通道 0/255 掩膜。
不把裂缝掩膜转换成检测框或多边形，避免细线与孔洞丢失。

原始文件只读，既有 processed 目录拒绝覆盖；重新处理时可使用 `--name 新名称`。
只质检已有数据可执行：

```bash
odp-validate apps/platform/configs/datasets/crackmap.yaml
```

WARNING 返回码为 1，ERROR 为 2。准备命令在没有 ERROR 时完成，风险提示仍保留在报告中。

## 本次数据检查结果

| 项目 | 结果 | 处理 |
|---|---|---|
| 官方划分 | train 84 / val 12 / test 24 | 保留原始划分 |
| 缺配对、坏文件、尺寸不一致 | 未发现 | 转换前检查，训练前再次核验 |
| 完全相同图片跨划分 | 未发现 | 按解码后的像素检查 |
| 图像和掩膜的 alpha 通道 | 原始 PNG 均为 RGBA | 显式转 RGB / L；alpha 不作为前景 |
| 训练/验证/测试前景占比 | 约 4.01% / 6.87% / 3.02% | 报告保留像素覆盖率 |
| 同前缀跨划分 | 13 组 | WARNING；可能同源裁剪，尚不能仅凭文件名证明泄漏 |

例如 `GOPR0315 (...)` 这类文件名可能指向同一段视频或原图的多个裁剪。当前资料
没有完整采集分组记录，因此不把推测当事实，也不擅自重划上游 split。
正式对外报告性能前，应确认这些分组并考虑按原图/视频划分。

新契约同时记录每个样本的图像 SHA256、掩膜 SHA256 和 split。
改图像、改掩膜、挪 split、添加/删除样本都会导致身份校验失败。

## 原始训练参数与两个模型配置

训练参数对照上游 [main.py](https://github.com/spiderforest/MixerCSeg/blob/7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d/main.py)：

| 参数 | 配置 |
|---|---|
| 输入尺寸 | 512×512 |
| train / test batch | 1 / 1 |
| epochs / seed | 50 / 42 |
| optimizer | AdamW |
| lr / min_lr | 5e-4 / 1e-6 |
| weight_decay | 0.01 |
| scheduler | mmengine PolyLR，保留其默认 power |
| loss | 0.87 × BCEWithLogits + 0.13 × Dice |
| 数据预处理 | RGB、cubic resize、normalize(0.5, 0.5)、掩膜阈值 127 |

数据已按上游代码处理 alpha，训练时仍采用它的 cubic+threshold 规则。
平台修正了上游用 test 集逐 epoch 选 best 的流程：用 val 选 best，test 保持独立。
输出使用跨平台 RunContext 命名，避免原脚本目录名中的冒号在 Windows 失效。

官方 checkpoint 与当前源码还有一个需要显式区分的结构差异：

- `mixercseg_train.yaml`：源码的 HOG bins=36，2,507,322 个参数，从头训练。
- `mixercseg_infer.yaml`：官方 CrackMap checkpoint 的 HOG bins=180，2,541,882 个参数。

官方权重已经通过 `strict=True` 结构匹配。加载错配置会报出 bins 差异，
不会使用 `strict=False` 跳过错误。若想训练 180-bin 变体，请复制训练配置并把
`hog_bins` 明确改成 180；其他原始训练参数保持不变。

```bash
# 完整训练配置，默认 50 epochs
odp-train --backend mixercseg --config mixercseg_train

# 快速检查一批训练 + 一批验证，审计明确记为 smoke
odp-train --backend mixercseg --epochs 1 --smoke-batches 1 --no-archive
```

本次已跑通一批真实训练、反向传播和一批验证；这不代表完成 50 epochs，
也不代表新的训练模型达到论文精度。本机是 CPU 版 PyTorch，PyTorch 扫描参考
实现能运行但训练较慢；完整实验宜在 CUDA 环境进行。

CUDA 环境还需按上游安装说明编译 SelectiveScan 扩展，单独安装 CUDA 版 torch
不会自动获得该内核。本适配保留其调用接口，缺少指定内核时走 PyTorch；
即使机器装有 CUDA 扩展，显式选择 CPU 也始终使用 CPU 参考实现。

上游使用 Python 3.10 / torch 2.1 / CUDA 11.8 等环境。本机使用当前 PyTorch
兼容路径，没有降级用户环境；模型结构和配方一致不等于逐位复现原始 CUDA 结果。

## D7 visualization 和推理

从上游 [checkpoints 文件夹](https://drive.google.com/drive/folders/13cfgbyYiBMcxdLLkQTB984RyDQOiJ1B4)
取得 CrackMap checkpoint，或使用以下命令下载到标准路径：

```bash
python -m gdown 1aPyHQvbN6mJm-ojqvBNwiLNp7V6WSw72 -O models/pretrained/checkpoint_CrackMap.pth
odp-infer --backend mixercseg --no-show --save
```

默认推理 CrackMap 的 24 张 test 图片。本次已用官方权重完成这条真实链路。
每次运行在 `runs/semantic_infer/<run_id>/` 生成：

- 标注叠加图片。
- `masks/`：原始图像尺寸的二值预测 PNG。
- `comparisons/`：原图、GT、预测、像素误差四联图。TP 绿、FP 红、FN 蓝。
- `odp_audit.json`：配置、上游版本、checkpoint SHA256、源、帧数与取消状态。

模型输出经 sigmoid 后再按固定 0.5 阈值处理，不沿用上游按每张图最大值
归一化 logits 的做法，避免全零除法和负值反转。只有同数据集的 images split
才自动匹配 GT；其他图/视频/摄像头源不猜测标注位置。

没有 GT 时输出原图、概率热力图和预测叠加。颜色、透明度、轮廓线见
`apps/platform/configs/runtime/mixercseg_pipeline.yaml`。

```bash
odp-infer --backend mixercseg --source path/to/image.png --no-show --save
python -m apps.desktop.main
```

桌面选 MixerCSeg 后会填入 CrackMap 权重与测试集路径；支持显示、停止与线程收尾。
桌面会从 checkpoint 的实际 HOG 维度选择结构，并严格加载。
`OutputSink` / hooks / CancelToken 和原有四级流水线继续复用。

模型权重、数据集和运行结果均被 Git 忽略，不上传私有仓库。

## Web 监控与待 GPU 完整实验

完整配置 `mixercseg_full.yaml` 保留 50 epochs、512×512、batch=1 和原始优化配方。
当前完整训练等待 GPU，不会由页面保存配置或刷新自动启动。
Web 端的 CUDA 门禁也不会自动回退到 CPU 执行完整训练。

监控和训练后的自动验收见 [Web 使用说明](../apps/web-backend/README.md)。
训练过程中每批原子写入进度、逐轮保存权重；正常结束后使用 best 权重在独立 test 集
按原图尺寸统计混淆矩阵、F1、前景 IoU、mIoU、precision 和 recall，并输出逐图 CSV、
损失曲线和掩膜对比图。固定阈值，不用 test 集选择权重或调整阈值。

新增可选 `cpu` extra 与 `scan_backend: cpu_jit`。它仅用 Numba 编译相同的逐步递推
和解析梯度，未开启 fastmath；前向和所有可训练输入梯度已与上游参考实现对齐测试。
默认训练继续使用 `reference` 配置，由上游分派器选择 CUDA 扩展；CPU JIT 用于本机
无 CUDA 时的单图推理验证，不改变上游 vendored 源码。

从本平台新训练的 36-bin 权重推理时，使用对应训练配置：

```bash
odp-infer --backend mixercseg --yaml apps/platform/configs/runtime/mixercseg_train.yaml --model runs/training/<run_id>/weights/best.pth --no-show --save
```
