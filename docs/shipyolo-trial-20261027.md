# ShipYOLO 短跑记录（2026-10-27）

用户指定 [ShipYOLO](https://github.com/Bilallbayrakdar/ShipYOLO)，检出的 HEAD 为
`ffba7f15ebdb84760af7a6b9ba0286babc9a12c1`。上游 README 的成绩使用其自身划分与评测方法，不能直接与本项目结果等同。
上游示例配置的 val 与 test 指向同一文件；本次保留项目中分组划分的训练 5638、验证 1052、测试 777 张，14 类顺序也使用现有标签对应顺序。
数据指纹 `32aa25208e93f3055f2dddcccf214d7422f62b759113b1a0644cbc02b344ab0b`；使用独立副本，未改写冻结数据。

## 适配与限制

- 仓库的 `run_train.sh` 引用缺失的 `train_hx.py`；使用仓库内的 `train.py`。
- 仓库未附可用预训练权重，从随机初始化开始。训练模型约 4990 万参数；512 输入、batch 8、4 workers、AMP、SGD，远程 RTX 4090。
- 保留源代码和修改于远端独立实验目录，使用单独虚拟环境；旧 PyTorch 版本不适配当前 4090 环境。
- 为新版运行时修复 NumPy 的 `np.int`、GPU 目标索引设备；按 README 使用 `deploy=False` 训练，并关闭 cuDNN benchmark 避免首批反向传播长时间卡住。
- 上游加载器在非训练时也会随机增强和翻转，独立测试前改为仅训练阶段增强；原始结果不能与修复后的测试混用。
- 加载器虽计算了 workers，实际固定为 1；短跑结束后改为使用计算值，但尚未重新测速，不能断言加速幅度。

## 实测

每轮约 5 分钟，4 轮训练计算完成，但第 4 轮验证绘图将 GPU 张量直接转 NumPy 而报错，导致该轮权重未保存。
前三轮权重保留在远端 `runs/shipyolo_experiment/runs/full4-nobench/weights/`。前三轮上游未执行验证；
虽然文件名为 `best.pt`，它只是第 3 轮检查点，**不是通过验证集选出的最佳权重**。

以第 3 轮检查点在 777 张测试图运行上游评估器，关闭测试增强：precision 2.94%、recall 1.07%、
mAP50 **1.30%**、mAP50–95 **0.28%**。报告已回传至本机 `runs/shipyolo_experiment/test_report.json`。
上游评估与现有 Ultralytics 的细节不同；短跑结果不构成充分训练的模型能力结论，也不能证明它优于当前精度方案。

之后修复绘图中的 GPU 张量转 CPU 问题，并使用第 3 轮权重完成 1052 张验证图和绘图，验证 mAP50 为 0.58%。
评测前未使用验证集选择检查点；该验证仅用于检查修复后的验证链路，不用于重新选择模型。

本次未将 ShipYOLO 集成到网页，也未回传 382 MB 的短跑权重。远端实例或其磁盘被释放时权重可能无法继续使用；
本机已保存报告和修复后的源代码压缩包 `runs/shipyolo_experiment/patched_source.tar.gz`。
下一步需要寻找可信预训练权重或增加训练预算，再进行同口径测试。
