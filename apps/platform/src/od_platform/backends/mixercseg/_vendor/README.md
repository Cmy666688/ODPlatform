# MixerCSeg 上游代码

来源：[spiderforest/MixerCSeg](https://github.com/spiderforest/MixerCSeg)，MIT 许可见 `LICENSE`。
固定提交：`7ad3fd6fd0e9bc60d1eb9c3b50292d876485dd1d`。
各文件的上游/适配后 SHA256 见 `UPSTREAM.json`。

这里保留上游模型结构及写法。只做包路径和运行环境适配：

- 绝对 `models` / `VMamba` 导入改为包内相对导入，避免污染宿主命名空间。
- timm 改用当前兼容导入路径；未使用的 mmcv 权重初始化移到方法内部。
- 原版只用 SS2D v2；未包含的 Mamba2 路径明确报错。
- CrossScan / CrossMerge 使用原文件中的 PyTorch autograd 实现，去掉导入时
  必须存在的 Triton decorator 和 CPU 不支持的 `torch.cuda.device` 上下文。
- SelectiveScan 保留上游参考实现。可用时沿用 CUDA 扩展，否则走 PyTorch。
  CPU 输入不调用 CUDA 扩展，指定的 CUDA 内核未安装时也明确回退到参考实现。

180-bin 官方 checkpoint 与 36-bin 源码的差异由外层 `model.py` 显式处理；
所有权重仍用 `strict=True` 加载，不允许缺失/错配参数静默通过。

Ruff/Mypy 不改写第三方源码；外层平台代码仍完整检查。
