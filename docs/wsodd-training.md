# WSODD 水面目标检测实验

## 当前状态

原始数据已接入、完整解码与质检已完成，Django / Vue / Streamlit 默认入口已切换到 WSODD。
2026-10-26 已切换至新远程实例，实测 NVIDIA GeForce RTX 4090 24GB、PyTorch 2.10.0+cu128、CUDA 矩阵计算通过。
完整实验 `c6e42c0be10949318a22cc23d3bcbf1a` 已完成 50/50 轮训练与独立测试，2026-10-27 已回传全部产物和 48 张对比图。
测试 mAP50 为 16.92%、mAP50–95 为 8.18%，效果待改善；详见 [验收记录](acceptance-20261027.md)。
训练、独立测试与推理不使用本机 CPU 回退。

## 数据与划分

- 原包：`data/raw/wsodd/WSODD USV_dataset.zip`，1,799,248,744 字节。
- 7,467 张 JPG、7,467 个同名 VOC XML，21,911 个目标框，共 14 类。
- 类别顺序固定为 boat、ship、person、bridge、rock、ball、platform、rubbish、mast、harbor、animal、buoy、tree、grass。
- 派生数据：`data/processed/wsodd/images/{train,val,test}` 与 `labels/{train,val,test}`。
- 训练 5,638 张，验证 1,052 张，测试 777 张，15 个采集分组。
- 按文件名中的相机与采集日分组；无时间信息时按 XML 源文件夹分组；相同 JPEG 文件合并其采集组。
- 用固定种子 42 搜索接近 70/15/15 且尽量覆盖类别的组划分，不随机打散相邻视频帧。
- 原包没有官方划分文件；采集日只能作为场景分组线索，不能证明所有场景彼此独立。
- 验证集无 tree，测试集无 animal；这些缺失类别不报告虚构的 AP。
- 已通过图像完整解码、标注尺寸、框范围、配对、跨集合相同 JPEG、分组交叉与 SHA256 身份检查。

数据身份指纹：`32aa25208e93f3055f2dddcccf214d7422f62b759113b1a0644cbc02b344ab0b`。
Manifest：`data/processed/wsodd/manifest.json`；质检：`runs/data_validation/wsodd/report.json`。
原件保持只读，预处理程序为 `python -m od_platform.data_pipeline.wsodd`，已有 processed 输出不会被覆盖。

## 完整训练配方

YOLOv8n 官方 COCO 预训练权重初始化，检测头适配 14 类；与此前 MixerCSeg 的网络、损失和像素指标分开。
默认 50 epochs、batch 16、640×640、AdamW、lr0=0.001、weight_decay=0.0005、seed=42。
使用项目原有 `YOLOTrainConfig` 生成引擎参数，patience=0，完整跑完配置轮次，启用 AMP。
训练完成后按 Ultralytics 验证集 fitness 选 best，test 不参与选模，独立计算 mAP50、mAP50–95、宏平均 P/R。
报告中的辅助 F1 为宏平均 P/R 的调和均值，与此前固定阈值像素分割 F1 不可直接比较。

最终应回传 best/last、训练曲线、逐类 CSV、PR 曲线、混淆矩阵、完整测试指标及最多 48 张 GT/预测对比图。
不设置业务精度承诺，验收状态为待人工评审。目标检测图片推理只使用该数据集的已完成训练权重。

## 远程执行

训练、独立测试与模型推理保持远程 CUDA 执行。本机仅做数据管理、网页和产物归档。
检测任务不依赖 MixerCSeg 的 selective-scan 内核。
万级文件合成一个受控 TAR 上传，远端只允许解包到 WSODD 派生目录及指定配置、初始化权重位置，开训前重新核验数据指纹。
网页提交前复用未改变大小和修改时间的图像摘要缓存，并重读检测标签；显式重做质检以及远端开训检查始终重读所有图像字节，不依赖该本机缓存。
SSH 重连复用原任务，不覆盖已启动的远程输入；只有产物回传完成才显示“已完成”。

原 CrackMap 与流程测试的实验现场保留在历史记录中，当前监控、模型选择和结果来源按所选数据集过滤。

## 多类别网页上传

Django 数据集管理现支持多类别 VOC / YOLO 目标检测压缩包和文件夹；上限 4 GiB，解压 12 GiB、100000 文件。
名称示例为“港区水域通航目标检测数据集”。已在浏览器实际上传 1,799,248,744 字节的 WSODD 原包，后台已完成独立转换与质检,识别 7467 张图像和 14 类目标。
新导入台账 ID：`ds_a8cb7385a5d7461e`；训练/验证/测试为 5284/1133/1050。通用导入按文件名前缀推断来源组,与前文 WSODD 专用预处理的非日期文件夹分组规则不同,因此划分不同。
原始上传、类别、数据身份与转换记录均保留;质检 WARNING 的类别覆盖限制保留在报告中。
该导入用于验证网页接入；本次正式训练使用前文已完整解码与质检的 WSODD 标准数据及其固定类别映射。

## 远程功能验收

2026-10-26 使用远程 RTX 4090、Ultralytics 8.4.183 完成合成三类别样本的 1 轮训练、独立 test、逐类 CSV、曲线及目标框对比图生成。
该小样本只验证链路,不代表 WSODD 精度。现场为 `.odp-meta/detection-gpu-smoke/cbc5f05367724d318ca3964b97cab340/job`,
本机证据为 `runs/wsodd_acceptance/gpu-smoke/`。
验收发现并修复新版 Ultralytics 字典损失格式、最终 best 验证重复回调问题。
远端已缓存官方预训练权重、AMP 检查权重和 Arial 绘图字体,独立测试所需 faster-coco-eval 已安装。
正式任务保留原始 50 轮配置与 WSODD 数据身份,不以合成样本替代。

## 来源

[WSODD 官方仓库](https://github.com/sunjiaen/WSODD)；用户提供的百度网盘原包。
[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) 与 [训练回调](https://docs.ultralytics.com/usage/callbacks/)。
