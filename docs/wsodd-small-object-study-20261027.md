# WSODD 小目标与类别不均衡对照（2026-10-27）

本轮只使用既有训练集和验证集选择方案。起点是 `605bb1d932834b93afdad2cb37d18e11` 的 YOLOv8s 最佳权重，SHA-256 为 `3c163833d3bb51df2b33326d794fdac87a1de0b4d4ca1f1d1eb0899af4b90bc2`，数据划分沿用既有指纹 `32aa25208e93f3055f2dddcccf214d7422f62b759113b1a0644cbc02b344ab0b`。没有拿测试集选择本轮参数；但此前迭代已多次使用该测试集，后续还需要新的独立数据检验泛化。

## 参考资料与针对性

- [WSODD 数据集论文](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2021.723336/full)：水面场景存在远距离、小尺寸和多尺度目标。[数据集仓库](https://github.com/sunjiaen/WSODD)主要提供数据链接及 VOC 到 COCO 的转换脚本；不能直接从中提取一套完整训练配置。
- [SAHI 论文](https://arxiv.org/abs/2202.06934)与[实现](https://github.com/Small-Object-Detection/SAHI)：使用重叠切片检测小目标，再与整图结果合并。下述实验是参考其思路的独立实现，不是对 SAHI 官方代码的复现。
- [Ultralytics 训练配置](https://docs.ultralytics.com/modes/train/)提供基于类别频率的 `cls_pw`，用于尝试处理类别不均衡。项目当前训练数据类别分布不均，例如 mast 训练集 33 个标注、验证集 281 个标注，划分差异可能影响泛化。

## 类别权重微调

远程 RTX 4090 上使用相同起始权重和数据，各额外微调 10 轮：YOLOv8s，960，batch 32，AdamW，初始学习率 0.0001，余弦衰减，RAM 缓存。分别尝试 `cls_pw=0.25` 和 `cls_pw=0`。训练和两次验证分别合计 486.28 秒及 496.58 秒；不包含上传等网络开销。

| 指标（完整验证集，Ultralytics） | 起始模型 | 类别权重 0.25 | 类别权重 0 |
| --- | ---: | ---: | ---: |
| mAP50 | 22.70% | 20.93% | 21.11% |
| mAP50–95 | 12.39% | 11.53% | 11.55% |
| boat AP50 | 54.13% | 51.54% | 53.01% |
| ship AP50 | 54.14% | 51.51% | 49.42% |

两组候选均低于起始模型，未替换默认权重。两组继续训练结果较接近，不能仅凭这次实验断言权重是下降原因。固定 seed 但训练 `deterministic=False`，运行间仍可能有随机差异。完整报告分别位于 `runs/accuracy_improvement/20261027/class_weight_report.json` 和 `no_class_weight_report.json`。

## 切片推理

对同一完整验证集的 1052 张图和 5847 个标注框，分别运行整图与整图加重叠切片（重叠 20%，输入 960，置信度下限 0.01，类内 IoU 0.7 NMS）。两种模式均使用相同 COCO 评估。切片大小是原始图像像素；COCO 小目标 AP 使用原图标注面积小于 32×32 像素，指标为 IoU 0.50–0.95 的平均精度。平均时间包含推理、切片和合并，不含读图，取决于当时硬件及批量参数。

| 验证结果 | 整图 | 1024 切片 + 整图 | 512 切片 + 整图 |
| --- | ---: | ---: | ---: |
| mAP50 | 19.22% | 17.11% | 15.44% |
| mAP50–95 | 10.34% | 8.46% | 7.35% |
| 小目标 AP | 1.62% | 3.05% | 2.86% |
| boat AP50 | 52.72% | 54.04% | 43.82% |
| ship AP50 | 51.32% | 37.21% | 23.66% |
| 每图时间 | 约 14–16 ms | 约 90 ms | 约 162 ms |

这组 COCO 对照使用的置信度下限等评估设置与上面的 Ultralytics 验证不同，绝对数字不能直接跨表比较。验证集中 tree 无标注，其 AP 不纳入平均值。完整类别结果和原始设置见 `runs/accuracy_improvement/20261027/slicing_full_report.json` 与 `slicing_512_report.json`。512 切片在此前港区样例图中补检了中央船只（ship 0.62），但同时出现重复框和误检，且整体验证指标下降，因此未部署为网页默认推理方式；也不能仅凭单张样例证明实际准确率提升。

可复现的对照脚本是 `scripts/compare_sliced_validation.py`，需要已安装 Ultralytics、PyTorch、torchvision、OpenCV 和 pycocotools 的 CUDA 环境。运行示例：

```bash
python scripts/compare_sliced_validation.py --data /path/to/dataset.yaml --weights /path/to/best.pt --count 1052 --tile-size 1024 --output /path/to/output
```

脚本保存验证集清单、两种预测、COCO 标注和指标，方便复查；使用其他数据集时 `--count` 大于样本数会使用全部样本。已完成静态检查及远程 GPU 实测。
