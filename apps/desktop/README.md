# apps/desktop — 桌面推理示例

使用 PySide6 复用 `apps/platform/` 引擎。模型与输入源选择、实时画面、进度、
停止和关闭窗口均在桌面端实现。引擎通过 `OutputSink` / `InferHooks` /
`CancelToken` 三个接缝与 UI 连接。

## 安装与运行

在仓库根目录执行：

```bash
pip install -e "./apps/platform[desktop]"
python -m apps.desktop.main
```

也可执行 `python apps/desktop/main.py`。需要桌面显示环境。

- 模型：填模型名、已有 `.pt` / `.onnx` 路径，或点击“选择权重”。已有权重优先
  从 `models/trained/`、`models/pretrained/` 查找；不存在的模型名沿用引擎下载行为。
- 输入：填 `0` 等摄像头号、图片/视频路径、图片目录或网络流 URL。
- 点击“开始”后，推理在 QThread 运行，主线程只处理 Qt UI。
- “停止”直接设置线程安全的取消 token，不排队到正在推理的 worker。
- 关闭窗口时先请求取消，保持 Qt 事件循环运行，在线程退出后完成关闭。
  实际停止耗时受当前模型计算及输入设备/网络读取耗时影响。

帧通过 Qt 信号送到主线程。`QtSignalSink` 使用独立的连续图像副本，避免底层
buffer 被复用；QObject 信号容器与 OutputSink 分开持有，避免 metaclass 冲突。
每次任务结束后会释放 worker 和 QThread，可再次启动。

桌面不额外保存标注图片/视频；引擎仍把审计 JSON 放到 `runs/<task>_infer/`，
日志由桌面入口装配，落到 `apps/platform/logging/desktop/`。

## 无显示环境验证

安装 `[dev,desktop]` 后执行：

```bash
python scripts/check_project.py
```

其中桌面回归使用 offscreen Qt 与假推理服务，覆盖图像副本、线程派发、取消、
失败收尾、重复运行和关闭窗口，不访问摄像头或网络模型。

线程生命周期遵循 [Qt QThread 文档](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html)。
