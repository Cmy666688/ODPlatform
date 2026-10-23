# frame_source

统一帧输入源抽象层(可插拔 · 多模态): 把任意输入源封装成同一套 `for frame in source` 接口。

## 特性
- 四种内置源: 摄像头 / 视频·网络流 / 单图 / 图片文件夹, 统一 `for frame in src` 迭代
- 配置驱动派发: 字符串糖("0"/"x.mp4"/"./imgs")或显式 config; 硬件源靠 config 类型选型
- 可插拔: 一行 @register_source 加源, 内置或第三方均不改本模块(开闭原则)
- 多模态 Frame: 主平面 image + 可选 depth / aux / intrinsics; modalities 能力发现
- 正交增强: ThreadedSource(后台采集)/ AsyncSource(异步接口)
- 上下文管理器 + fail-fast 配置校验, 资源不漏

## 依赖
opencv-python, numpy, pydantic>=2.0, Python>=3.10

## 快速开始
    from od_platform.frame_source import create_frame_source
    with create_frame_source("0") as src:      # "0"/"x.mp4"/"rtsp://..."/"./imgs"
        for frame in src:
            do_something(frame.image)

## 显式选型(硬件 / 高帧率 / 内存源)
    from od_platform.frame_source import create_frame_source, CameraConfig
    with create_frame_source(CameraConfig(fps=90, backend="msmf")) as src:
        ...

## 加一种源(不改本模块)
    from od_platform.frame_source import FrameSource, SourceConfig, register_source

    class MyConfig(SourceConfig): ...
    @register_source(MyConfig)
    class MySource(FrameSource): ...
    # import 到你的源模块即接入; 或用 entry-points + load_plugins() 自动发现

## 独立性
无宿主项目依赖, 可整包拷贝至任意项目。第三方源可独立扩展。
