"""SSH 大文件中断后接续已确认的字节,最终核对远程摘要。"""
import hashlib
import io
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from od_web.remote import Connection


@pytest.mark.parametrize("corrupt", [False, True])
def test_interrupted_transfer_resumes_and_rejects_corrupt_prefix(tmp_path, monkeypatch, corrupt):
    source = tmp_path / "dataset.tar"; source.write_bytes(b"frame0123" * (1024 * 1024))
    remote = {}; failed = False; target = "/gpu/dataset.tar"
    class Stream(io.BytesIO):
        def __init__(self, name, mode):
            self.name, self.mode = name, mode
            if "w" not in mode and name not in remote: raise FileNotFoundError(name)
            super().__init__(b"" if "w" in mode else remote[name])
        def set_pipelined(self, value): assert value
        def write(self, value):
            nonlocal failed
            result = super().write(value.encode() if isinstance(value, str) else value)
            if self.name.endswith(".upload") and self.tell() >= 3 * 1024**2 and not failed:
                failed = True; raise OSError("simulated SSH disconnect")
            return result
        def close(self):
            if "w" in self.mode or "+" in self.mode: remote[self.name] = self.getvalue()
            super().close()
    class Transfer:
        def open(self, name, mode="r"): return Stream(name, mode)
        def stat(self, name):
            if name not in remote: raise FileNotFoundError(name)
            return SimpleNamespace(st_size=len(remote[name]))
        def posix_rename(self, old, new): remote[new] = remote.pop(old)
    class Client:
        def exec_command(self, command, **kwargs):
            output = io.BytesIO((hashlib.sha256(remote[target + ".upload"]).hexdigest() + " file\n").encode())
            output.channel = SimpleNamespace(recv_exit_status=lambda: 0)
            return None, output, io.BytesIO()
    connection = object.__new__(Connection); connection.sftp = Transfer(); connection.client = Client()
    monkeypatch.setattr(connection, "mkdir", lambda _: None)
    with pytest.raises(OSError, match="disconnect"): connection.put(source, PurePosixPath(target))
    partial = len(remote[target + ".upload"]); assert 0 < partial < source.stat().st_size
    if corrupt: remote[target + ".upload"] = b"X" + remote[target + ".upload"][1:]
    progress = []
    def send(): connection.put(source, PurePosixPath(target), on_progress=lambda done, total: progress.append(done))
    if corrupt:
        with pytest.raises(OSError, match="摘要不一致"): send()
        assert target not in remote
        assert remote[target + ".upload.identity"] == b""
        send()
    else: send()
    assert progress[0] == partial
    assert remote[target] == source.read_bytes()
    assert remote[target + ".sha256"].decode() == hashlib.sha256(source.read_bytes()).hexdigest()
