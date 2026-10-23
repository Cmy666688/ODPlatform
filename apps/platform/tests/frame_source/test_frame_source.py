#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  : test_frame_source.py
# @Author    : 雨霓同学
# @Project   : ODPlatform / frame_source
# @Function  : frame_source 自测 —— 注册表 + 多模态 + 四源 + 扩展性实证
"""frame_source 自测。

双入口:
    - 独立运行: python tests/frame_source/test_frame_source.py
    - pytest:   pytest apps/platform/tests/frame_source/test_frame_source.py
摄像头/视频不可用时相关项自动跳过。

注: 相对讲义沙箱版仅改一处 import 前缀(frame_source → od_platform.frame_source),
    适配本仓库 src 布局; 其余代码逐字保留, 并追加 pytest 入口函数。
"""
from __future__ import annotations

import asyncio
import tempfile
import shutil
from pathlib import Path

import numpy as np
import cv2
from pydantic import ValidationError

from od_platform.frame_source import (
    create_frame_source, create_threaded_source, create_async_source,
    CameraConfig, VideoConfig, ImageConfig, ImageFolderConfig, SourceConfig,
    Frame, FrameInfo, CameraIntrinsics, SourceType,
    FrameSource, register_source, build_from_config, build_from_string, list_sources,
    ThreadedSource, BufferStrategy, AsyncSource, ReconnectingSource, EndReason,
)

# ── 迷你测试框架 ──
_passed = 0
_failed = 0
_skipped = 0

def check(name: str, cond: bool) -> None:
    global _passed, _failed
    mark = "✅" if cond else "❌"
    print(f"  {mark} {name}")
    if cond: _passed += 1
    else:    _failed += 1

def skip(name: str, reason: str) -> None:
    global _skipped
    print(f"  ⏭️  {name}(跳过: {reason})")
    _skipped += 1

def expect_raises(name: str, exc_types, fn) -> None:
    global _passed, _failed
    try:
        fn()
        print(f"  ❌ {name}(未抛异常)")
        _failed += 1
    except exc_types:
        print(f"  ✅ {name}")
        _passed += 1
    except Exception as e:
        print(f"  ❌ {name}(抛了非预期异常: {type(e).__name__}: {e})")
        _failed += 1


# ── 一个会"断线一次再恢复"的假源, 用于验证重连 ──
class FlakyConfig(SourceConfig):
    fail_after: int = 3      # 第一次会话跑到第几帧就"断线"
    total:      int = 6      # 重连后正常跑完多少帧

class FlakySource(FrameSource):
    def __init__(self, config: FlakyConfig):
        super().__init__(config)
        self._fail_after = config.fail_after
        self._total = config.total
        self._i = 0
        self._session = 0    # 第几次 open(重连会 +1)

    def open(self):
        self._i = 0
        self._session += 1
        self._end_reason = EndReason.RUNNING
        return True

    def read(self):
        limit = self._fail_after if self._session == 1 else self._total
        if self._i >= limit:
            # 第一次会话到 limit 就"断线", 触发重连; 之后会话正常读完
            self._end_reason = (EndReason.DISCONNECTED if self._session == 1
                                else EndReason.COMPLETED)
            return None
        img = np.zeros((4, 4, 3), np.uint8)
        info = FrameInfo(width=4, height=4, source_type="flaky",
                         source_path=self.source_path, frame_index=self._i)
        self._i += 1
        return Frame(image=img, info=info)

    def close(self): pass
    def get_source_type(self): return "flaky"


# ── 准备临时素材 ──
def make_fixtures(root: Path):
    # 单张图
    img_path = root / "one.jpg"
    cv2.imwrite(str(img_path), np.full((48, 64, 3), 127, np.uint8))
    # 图片文件夹(6 张, 含一张大小写后缀)
    folder = root / "imgs"; folder.mkdir()
    for i in range(5):
        cv2.imwrite(str(folder / f"{i:02d}.jpg"), np.full((48, 64, 3), i * 40, np.uint8))
    cv2.imwrite(str(folder / "05.PNG"), np.full((48, 64, 3), 200, np.uint8))   # 大写后缀也要认
    # 短视频(可能因编码器不可用而失败)
    vid_path = root / "clip.mp4"
    vw = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
    ok_video = vw.isOpened()
    if ok_video:
        for i in range(10):
            vw.write(np.full((48, 64, 3), i * 20, np.uint8))
        vw.release()
    else:
        vw.release()
    return img_path, folder, (vid_path if ok_video else None)


# ── 一个包外自定义源, 用于扩展性实证(config-only 内存源)──
class SyntheticConfig(SourceConfig):
    n_frames: int = 5
    @property
    def identity(self) -> str:
        return f"synthetic:{self.n_frames}"

@register_source(SyntheticConfig)     # 无 str_matcher: 只能靠 config 类型派发
class SyntheticSource(FrameSource):
    def __init__(self, config: SyntheticConfig):
        super().__init__(config)
        self._n = config.n_frames; self._i = 0
    def open(self):
        self._i = 0; self._frame_index = 0; return True
    def read(self):
        if self._i >= self._n: return None
        img = np.full((16, 16, 3), self._i, np.uint8)
        info = FrameInfo(width=16, height=16, source_type="synthetic",
                         source_path=self.source_path, frame_index=self._i, total_frames=self._n)
        self._i += 1; self._frame_index = self._i
        return Frame(image=img, info=info)
    def close(self): pass
    def get_source_type(self): return "synthetic"


def main():
    tmp = Path(tempfile.mkdtemp(prefix="fs_test_"))
    try:
        img_path, folder, vid_path = make_fixtures(tmp)

        print("\n=== 1. 多模态数据类型 ===")
        # 断言 Frame 带 depth/aux/intrinsics 且能用
        f = Frame(image=np.zeros((4, 4, 3), np.uint8),
                  info=FrameInfo(width=4, height=4, source_type=SourceType.IMAGE, source_path="x"))
        check("Frame 有 depth 字段且默认 None",       f.depth is None)
        check("Frame 有 aux 字段且默认空 dict",         f.aux == {})
        check("Frame 有 intrinsics 字段且默认 None",    f.intrinsics is None)
        check("无深度时 has_depth 为 False",           f.has_depth is False)
        fd = Frame(image=np.zeros((4, 4, 3), np.uint8), depth=np.zeros((4, 4), np.uint16),
                   info=f.info)
        check("有深度时 has_depth 为 True",            fd.has_depth is True)
        intr = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240, depth_scale=0.001)
        check("CameraIntrinsics 字段可读",             intr.fx == 600 and intr.depth_scale == 0.001)
        check("Frame.width 便捷属性",                  f.width == 4)

        print("\n=== 2. 注册表 ===")
        reg = list_sources()
        check("list_sources 含四个内置源",
              {"CameraConfig", "VideoConfig", "ImageConfig", "ImageFolderConfig"} <= set(reg))
        check("list_sources 含实证注册的 SyntheticConfig", "SyntheticConfig" in reg)
        # 配置类型派发
        check("build_from_config(ImageConfig) → ImageSource",
              type(build_from_config(ImageConfig(path=str(img_path)))).__name__ == "ImageSource")
        check("build_from_config(CameraConfig) → CameraSource",
              type(build_from_config(CameraConfig())).__name__ == "CameraSource")
        # 字符串糖派发
        check("build_from_string('0') → CameraSource",
              type(build_from_string("0")).__name__ == "CameraSource")
        check("build_from_string(图片路径) → ImageSource",
              type(build_from_string(str(img_path))).__name__ == "ImageSource")
        check("build_from_string(文件夹路径) → ImageFolderSource",
              type(build_from_string(str(folder))).__name__ == "ImageFolderSource")

        print("\n=== 3. 派发的失败路径(fail-fast)===")
        class _UnregConfig(SourceConfig): pass
        expect_raises("未注册的配置类型 → ValueError", ValueError,
                      lambda: build_from_config(_UnregConfig()))
        expect_raises("无法识别的字符串 → ValueError", ValueError,
                      lambda: build_from_string("nonsense.xyz"))
        expect_raises("build_from_config 收到非 config → TypeError", TypeError,
                      lambda: build_from_config("0"))
        expect_raises("视频路径不存在 → 构造 config 即抛", ValidationError,
                      lambda: VideoConfig(path="/no/such/file.mp4"))
        expect_raises("字符串糖遇不存在路径 → ValueError(包装)", ValueError,
                      lambda: create_frame_source("/no/such/file.mp4"))
        expect_raises("config 字段拼错 → ValidationError(extra=forbid)", ValidationError,
                      lambda: CameraConfig(fpss=90))
        expect_raises("图片源拒绝 URL → ValidationError", ValidationError,
                      lambda: ImageConfig(path="http://x/a.jpg"))

        print("\n=== 4. 能力发现 modalities ===")
        with create_frame_source(str(img_path)) as src:
            check("彩色源 modalities == {'image'}", src.modalities == frozenset({"image"}))
        # 深度能力靠源覆盖(此处用一个临时深度源验证覆盖点有效)
        class _DepthCfg(SourceConfig): pass
        @register_source(_DepthCfg)
        class _DepthSrc(FrameSource):
            def open(self): return True
            def read(self): return None
            def close(self): pass
            def get_source_type(self): return "depth"
            @property
            def modalities(self): return frozenset({"image", "depth"})
        check("深度源可覆盖 modalities 含 depth",
              "depth" in build_from_config(_DepthCfg()).modalities)

        print("\n=== 5. 四源端到端 ===")
        # 单图
        with create_frame_source(ImageConfig(path=str(img_path))) as src:
            frames = list(src)
            check("单图产 1 帧", len(frames) == 1)
            check("单图帧 image 形状正确", frames[0].image.shape == (48, 64, 3))
            check("单图 source_type=IMAGE", frames[0].info.source_type == SourceType.IMAGE)
        # 文件夹
        with create_frame_source(ImageFolderConfig(path=str(folder))) as src:
            frames = list(src)
            check("文件夹产 6 帧(含大写后缀 PNG)", len(frames) == 6)
            check("文件夹帧序稳定(frame_index 递增)",
                  [fr.info.frame_index for fr in frames] == list(range(6)))
        # 视频
        if vid_path:
            with create_frame_source(VideoConfig(path=str(vid_path))) as src:
                frames = list(src)
                check("视频产约 10 帧", 9 <= len(frames) <= 10)
                check("视频有 fps 元数据", frames[0].info.fps is not None)
                check("本地视频 seekable=True", src.seekable is True)
        else:
            skip("视频端到端", "VideoWriter 编码器不可用")
        # 摄像头(设备可能不存在)
        cam = create_frame_source("0")
        if cam.open():
            fr = cam.read(); cam.close()
            check("摄像头能读到一帧", fr is not None)
        else:
            cam.close(); skip("摄像头端到端", "无可用摄像头设备")

        print("\n=== 6. seek 与 stride ===")
        # 文件夹 seek
        fsrc = create_frame_source(ImageFolderConfig(path=str(folder)))
        fsrc.open()
        check("文件夹 seekable=True", fsrc.seekable is True)
        check("文件夹 seek(frame=3) 成功", fsrc.seek(frame=3) is True)
        check("seek 后读到第 3 帧", fsrc.read().info.frame_index == 3)
        fsrc.close()
        # 文件夹 stride
        with create_frame_source(ImageFolderConfig(path=str(folder)), stride=3) as src:
            idxs = [fr.info.frame_index for fr in src]
            check("stride=3 只产 0,3 两帧(零 IO 跳采)", idxs == [0, 3])
        # 视频 stride
        if vid_path:
            with create_frame_source(VideoConfig(path=str(vid_path)), stride=5) as src:
                got = list(src)
                check("视频 stride=5 产约 2 帧", 1 <= len(got) <= 2)
        else:
            skip("视频 stride", "无视频")
        # 摄像头 stride 应被忽略(不报错)
        cam2 = create_frame_source("0", stride=3)
        check("摄像头 stride 被忽略(stride 仍为 1)", cam2.stride == 1)
        cam2.close()

        print("\n=== 7. 重新 open 复位 + 迭代器单次 ===")
        src = create_frame_source(ImageFolderConfig(path=str(folder)))
        src.open(); n1 = len(list(src))
        src.open(); n2 = len(list(src))          # 重新 open 应能重头再来
        check("重新 open 后能重头迭代", n1 == n2 == 6)
        src.close()
        # 迭代器单次: 不重新 open, 第二次迭代应为空
        src2 = create_frame_source(ImageConfig(path=str(img_path)))
        src2.open()
        first = len(list(src2)); second = len(list(src2))
        check("同一 open 内迭代一次即耗尽", first == 1 and second == 0)
        src2.close()

        print("\n=== 8. wrappers: 线程 / 异步 ===")
        # 线程(用 BOUNDED 保证不丢帧, 便于精确计数)
        with create_threaded_source(SyntheticConfig(n_frames=10),
                                    strategy=BufferStrategy.BOUNDED, maxsize=32) as src:
            n = len(list(src))
            check("ThreadedSource(BOUNDED) 不丢帧, 产 10 帧", n == 10)
            check("ThreadedSource 透传 modalities", src.modalities == frozenset({"image"}))
        # 异步
        async def _run_async():
            cnt = 0
            async with create_async_source(SyntheticConfig(n_frames=5)) as src:
                async for _ in src:
                    cnt += 1
            return cnt
        check("AsyncSource async for 产 5 帧", asyncio.run(_run_async()) == 5)
        check("AsyncSource 不是 FrameSource 子类(守 LSP)",
              not issubclass(AsyncSource, FrameSource))

        print("\n=== 9. 扩展性实证(不改 frame_source 加源)===")
        with create_frame_source(SyntheticConfig(n_frames=3)) as src:
            got = list(src)
            check("包外 config-only 源可被 create_frame_source 派发", len(got) == 3)
            check("包外源可用第三方 source_type 字符串", got[0].info.source_type == "synthetic")
            check("内存源(无字符串形态)正常工作", got[0].image.shape == (16, 16, 3))

        print("\n=== 10. 错误处理健壮性 ===")
        # 文件夹里放一张坏图, 应跳过而非崩溃
        bad = folder / "broken.jpg"; bad.write_bytes(b"not an image")
        with create_frame_source(ImageFolderConfig(path=str(folder))) as src:
            frames = list(src)     # 坏图被跳过, 其余正常
            check("坏图被跳过, 不影响其余帧", len(frames) == 6)

        print("\n=== 11. 断线重连 ===")
        # 用极短退避让测试跑得快
        rc = ReconnectingSource(FlakySource(FlakyConfig(fail_after=3, total=6)),
                                backoff_initial=0.01, backoff_max=0.05, max_retries=5)
        with rc as src:
            got = list(src)      # 断线前 3 帧 + 重连后 6 帧 = 9 帧, 且不 hang
        check("重连拿到断线前+重连后的全部帧", len(got) == 3 + 6)
        check("重连后干净结束(end_reason=COMPLETED)", src.end_reason == EndReason.COMPLETED)

        # 附带验证 end_reason 语义本身
        img_src = create_frame_source(ImageConfig(path=str(img_path)))
        img_src.open(); _ = list(img_src)
        check("单图读完 end_reason=COMPLETED", img_src.end_reason == EndReason.COMPLETED)
        img_src.close()

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'='*44}")
    print(f"通过 {_passed}  失败 {_failed}  跳过 {_skipped}")
    print(f"{'='*44}")
    return 0 if _failed == 0 else 1


def test_frame_source_selftest():
    """pytest 入口: 完整跑一遍自测, 任何 check 失败 → main() 返回 1 → 本断言失败。"""
    assert main() == 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
