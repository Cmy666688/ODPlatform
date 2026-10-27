"""CLI 与服务共享的节流进度,回调不负责业务判断。"""
import logging
import time


class Progress:
    def __init__(self, phase, callback=None):
        self.phase, self.callback, self.last = phase, callback, 0.0

    def __call__(self, done, total):
        now = time.monotonic()
        if done not in (0, total) and now - self.last < 0.5: return
        self.last = now
        logging.getLogger(__name__).info("%s: %s / %s", self.phase, done, total if total else "待统计")
        if self.callback: self.callback(self.phase, done, total)
