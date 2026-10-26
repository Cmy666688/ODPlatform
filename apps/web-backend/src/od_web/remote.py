"""SSH GPU 执行与回传;网页不在本机回退到 CPU。连接配置不包含口令。"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time

from od_platform.common import paths
from od_platform.common.json_io import write_json

_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
TERMINAL = ("completed", "failed", "cancelled")


class RemoteRejected(RuntimeError):
    """远端已明确拒绝任务,区别于未收到启动确认的网络异常。"""


def utc_now(): return datetime.now(timezone.utc).isoformat()


@dataclass
class RemoteConfig:
    host: str
    port: int
    username: str
    root: str
    python: str
    key_file: str
    known_hosts: str
    extra_pythonpath: str = ""

    @classmethod
    def load(cls):
        filename = paths.META_DIR / "web" / "remote.json"
        if not filename.is_file(): raise RuntimeError("远程 GPU 尚未配置,请运行 python -m od_web.remote_setup")
        result = cls(**json.loads(filename.read_text(encoding="utf-8")))
        if not PurePosixPath(result.root).is_absolute() or ".." in PurePosixPath(result.root).parts:
            raise ValueError("远程工作目录必须是绝对路径")
        return result


class Connection:
    def __init__(self, config=None):
        import paramiko
        self.config = config or RemoteConfig.load()
        self.client = paramiko.SSHClient()
        self.client.load_host_keys(self.config.known_hosts)
        self.client.set_missing_host_key_policy(paramiko.RejectPolicy())
        try:
            self.client.connect(self.config.host, port=self.config.port, username=self.config.username,
                                key_filename=self.config.key_file, timeout=8, auth_timeout=8, banner_timeout=8,
                                look_for_keys=False, allow_agent=False)
            self.client.get_transport().set_keepalive(20)
            self.sftp = paramiko.SFTPClient.from_transport(self.client.get_transport(), window_size=16*1024**2, max_packet_size=128*1024)
            self.sftp.get_channel().settimeout(60)
        except Exception:
            self.client.close(); raise

    def close(self): self.client.close()

    def command(self, arguments: list[str]) -> dict:
        cfg = self.config
        pythonpath = ":".join(filter(None, [cfg.root + "/apps/platform/src", cfg.root + "/apps/web-backend/src", cfg.extra_pythonpath]))
        command = (f"cd {shlex.quote(cfg.root)} && env PYTHONUTF8=1 PYTHONUNBUFFERED=1 "
                   f"PYTHONPATH={shlex.quote(pythonpath)} {shlex.quote(cfg.python)} -m od_web.remote_agent "
                   + " ".join(shlex.quote(value) for value in arguments))
        _, stdout, stderr = self.client.exec_command(command, timeout=60)
        # 同时读取两路输出,避免第三方模型日志填满 SSH 窗口。
        channel = stdout.channel; out = bytearray(); err = bytearray()
        deadline = time.monotonic() + 60
        while not channel.exit_status_ready() or channel.recv_ready() or channel.recv_stderr_ready():
            if time.monotonic() > deadline:
                channel.close(); raise TimeoutError("远程命令超时")
            if channel.recv_ready(): out.extend(channel.recv(65536))
            if channel.recv_stderr_ready(): err.extend(channel.recv_stderr(65536))
            time.sleep(0.02)
        if channel.recv_exit_status(): raise RemoteRejected(err.decode("utf-8", "replace")[-1600:])
        lines = out.decode("utf-8", "replace").splitlines()
        for line in reversed(lines):
            if line.startswith("ODP_JSON="):
                result: dict = json.loads(line[9:]); return result
        raise OSError("远程 GPU 返回了无效响应,等待重新确认任务状态")

    def mkdir(self, path: PurePosixPath):
        if str(path) == "/": return
        try: self.sftp.stat(str(path))
        except FileNotFoundError:
            self.mkdir(path.parent); self.sftp.mkdir(str(path))

    def put(self, source: Path, target: PurePosixPath, *, on_progress=None):
        self.mkdir(target.parent)
        # 输入按内容核验,重连不重复传输相同文件。
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            while chunk := stream.read(1024 * 1024): digest.update(chunk)
        checksum = digest.hexdigest()
        marker = str(target) + ".sha256"
        try:
            if self.sftp.stat(str(target)).st_size == source.stat().st_size:
                with self.sftp.open(marker) as stream:
                    if stream.read().decode() == checksum: return
        except FileNotFoundError: pass
        temporary = str(target) + ".upload"
        total = source.stat().st_size
        if total < 8 * 1024**2:
            self.sftp.put(str(source), temporary)
        else:
            # 仅恢复与当前完整内容摘要匹配的临时文件;SSH 中断后不重传整个大包。
            identity = temporary + ".identity"; offset = 0
            try:
                with self.sftp.open(identity) as stream: previous = stream.read().decode()
                size = self.sftp.stat(temporary).st_size
                if previous == checksum and 0 <= size <= total: offset = size
            except FileNotFoundError: pass
            with self.sftp.open(identity, "w") as stream: stream.write(checksum)
            # 大文件走连续 SSH 字节流,避免高延迟网络中逐个 SFTP 写请求的开销。
            stdin, stdout, stderr = self.client.exec_command("cat " + (">> " if offset else "> ") + shlex.quote(temporary), timeout=60)
            try:
                with source.open("rb") as local:
                    local.seek(offset)
                    if on_progress: on_progress(offset, total)
                    reported = time.monotonic()
                    while chunk := local.read(1024 * 1024):
                        stdin.write(chunk); offset += len(chunk)
                        if on_progress and time.monotonic() - reported >= 2:
                            on_progress(offset, total); reported = time.monotonic()
                stdin.flush(); stdin.channel.shutdown_write()
                if stdout.channel.recv_exit_status(): raise OSError("远程文件写入失败: " + stderr.read().decode("utf-8", "replace")[-300:])
            finally: stdin.close()
            if self.sftp.stat(temporary).st_size != total: raise OSError("远程文件大小不完整,等待继续传输")
            _, stdout, stderr = self.client.exec_command("sha256sum " + shlex.quote(temporary), timeout=180)
            actual = stdout.read().decode().split(); error = stderr.read().decode("utf-8", "replace")
            if stdout.channel.recv_exit_status() or not actual: raise OSError("远程文件校验未完成: " + error[-300:])
            if actual[0] != checksum:
                with self.sftp.open(identity, "w") as stream: stream.write("")
                raise OSError("远程文件摘要不一致,下次从头传输")
            if on_progress: on_progress(total, total)
        self.sftp.posix_rename(temporary, str(target))
        with self.sftp.open(marker, "w") as stream: stream.write(checksum)

    def put_json(self, value: dict, target: PurePosixPath):
        self.mkdir(target.parent)
        with self.sftp.open(str(target) + ".upload", "w") as stream:
            stream.write(json.dumps(value, ensure_ascii=False))
        self.sftp.posix_rename(str(target) + ".upload", str(target))

    def get_json(self, source: PurePosixPath):
        with self.sftp.open(str(source)) as stream: return json.loads(stream.read())

    def get(self, source: PurePosixPath, target: Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download")
        self.sftp.get(str(source), str(temporary)); temporary.replace(target)

    def tree(self, source: PurePosixPath, target: Path):
        for entry in self.sftp.listdir_attr(str(source)):
            if entry.filename in (".", "..") or "/" in entry.filename or "\\" in entry.filename:
                raise ValueError("非法远程产物名")
            if stat.S_ISDIR(entry.st_mode): self.tree(source / entry.filename, target / entry.filename)
            elif stat.S_ISREG(entry.st_mode): self.get(source / entry.filename, target / entry.filename)


def hardware(*, force=False) -> dict:
    # API 高频刷新复用短缓存;缓存过期后断线不会继续报告 GPU 就绪。
    with _CACHE_LOCK:
        key = str(paths.META_DIR)
        if not force and _CACHE.get("key") == key and time.monotonic() - _CACHE.get("time", 0) < 5:
            return dict(_CACHE["value"])
        result = {"execution": "remote", "connected": False, "cuda_available": False,
                  "device": "远程 GPU 未连接", "torch": "—", "cpu_percent": None,
                  "memory_percent": None, "memory_gb": None, "memory_used_gb": None,
                  "gpu_percent": None, "gpu_memory_used_gb": None, "gpu_memory_gb": None}
        connection = None
        try:
            connection = Connection(); result.update(connection.command(["hardware"]))
            result["host"] = connection.config.host
            result["message"] = "远程 GPU 已连接,训练、独立测试和图片推理均在远程执行。" if result["cuda_available"] else "远程 CUDA 或扫描内核未就绪,请检查 GPU 环境。"
        except Exception as error:
            result["message"] = "远程 GPU 不可用: " + str(error)
        finally:
            if connection is not None: connection.close()
        result["sampled_at"] = utc_now()
        _CACHE.update(key=key, time=time.monotonic(), value=result)
        return dict(result)


def relative_file(value: str, prefix: str | None = None) -> PurePosixPath:
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in value or ":" in value or not relative.parts:
        raise ValueError("非法远程产物路径")
    if prefix and relative.parts[0] != prefix: raise ValueError("产物目录超出允许范围")
    return relative


def prepare(connection: Connection, directory: Path, request: dict):
    from od_platform.common.semantic import load_semantic_dataset
    remote = PurePosixPath(connection.config.root)
    document = {**request, "config": {**request["config"], "device": "cuda", "scan_backend": "reference"}, "require_cuda": True}
    if request["kind"] == "training":
        yaml_path = paths.dataset_yaml_path(request["config"]["data"])
        if request["config"].get("backend") == "yolo":
            import yaml
            dataset = yaml.safe_load(yaml_path.read_text(encoding="utf-8")); root = paths.ROOT_DIR / dataset["path"]
            # 约 1.8 GB、上万文件使用单个受控包传输,避免逐文件往返。
            bundle = directory / "dataset.tar"
            targets = [yaml_path, *sorted(root.rglob("*.jpg")), *sorted(root.rglob("*.txt")), paths.ROOT_DIR / dataset["manifest_path"]]
            pretrained = paths.PRETRAINED_MODELS_DIR / request["config"]["architecture"]
            if pretrained.is_file(): targets.append(pretrained)
            if not bundle.exists():
                with tarfile.open(bundle.with_suffix(".building"), "w") as archive:
                    for source in targets:
                        if source.is_symlink() or not source.resolve().is_relative_to(paths.ROOT_DIR.resolve()): raise ValueError("检测数据超出项目目录")
                        archive.add(source, arcname=source.relative_to(paths.ROOT_DIR).as_posix())
                bundle.with_suffix(".building").replace(bundle)
            def progress(done, total):
                state_file = directory / "status.json"
                state = json.loads(state_file.read_text(encoding="utf-8"))
                state.update(upload_bytes=done, upload_total_bytes=total, phase="uploading")
                write_json(state_file, state)
            connection.put(bundle, remote / "runs/web_jobs" / directory.name / "dataset.tar", on_progress=progress)
            connection.put_json(document, remote / "runs/web_jobs" / directory.name / "request.json")
            connection.command(["unpack", directory.name]); targets = []
        else:
            dataset, root = load_semantic_dataset(yaml_path)
            targets = [yaml_path, *sorted(root.rglob("*.png")), paths.ROOT_DIR / dataset["manifest_path"]]
        for source in targets:
            if source.is_symlink() or not source.resolve().is_relative_to(paths.ROOT_DIR.resolve()):
                raise ValueError("数据集输入超出项目目录")
            relative = source.relative_to(paths.ROOT_DIR).as_posix()
            connection.put(source, remote / relative)
    else:
        source = Path(request["source"])
        target = remote / "runs/web_uploads" / source.name
        connection.put(source, target); document["source"] = str(target)
        weight = Path(request["config"]["model"])
        target = remote / "models/web" / (directory.name + weight.suffix)
        connection.put(weight, target); document["config"]["model"] = str(target)
    remote_job = remote / "runs/web_jobs" / directory.name
    connection.put_json(document, remote_job / "request.json")
    try: connection.sftp.stat(str(remote_job / "status.json"))
    except FileNotFoundError:
        initial = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        initial.pop("remote_managed", None); initial["execution"] = "local"
        connection.put_json(initial, remote_job / "status.json")
    return remote_job


def start_bridge(directory: Path):
    import os
    with (directory / "bridge.log").open("ab") as log:
        process = subprocess.Popen([sys.executable, "-m", "od_web.remote", "--job-dir", str(directory)],
                                   cwd=paths.ROOT_DIR, env=dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1"),
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    write_json(directory / "bridge.json", {"pid": process.pid, "created_at": utc_now()})


def ensure_bridge(directory: Path, state: dict):
    import psutil
    if not state.get("remote_managed") or state["status"] in TERMINAL: return
    marker = directory / "bridge.json"
    if marker.exists():
        pid = json.loads(marker.read_text(encoding="utf-8"))["pid"]
        if psutil.pid_exists(pid): return
    elif (datetime.now(timezone.utc) - datetime.fromisoformat(state["created_at"])).total_seconds() < 10:
        return
    with _CACHE_LOCK:
        if marker.exists():
            pid = json.loads(marker.read_text(encoding="utf-8"))["pid"]
            if psutil.pid_exists(pid): return
        start_bridge(directory)


def bridge(directory: Path):
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    config = RemoteConfig.load(); remote = PurePosixPath(config.root)
    connection = None; prepared = False; launched = False
    while True:
        state = json.loads((directory / "status.json").read_text(encoding="utf-8"))
        if state["status"] in TERMINAL: break
        try:
            if connection is None: connection = Connection(config)
            if not prepared:
                remote_job = remote / "runs/web_jobs" / directory.name
                try:
                    connection.get_json(remote_job / "launch.json")
                    # 恢复同步时不改写已经开始运行的远程输入。
                    prepared = True; launched = True
                except FileNotFoundError:
                    state.update(phase="uploading", connection_status="connected", sync_error=None)
                    write_json(directory / "status.json", state)
                    remote_job = prepare(connection, directory, request); prepared = True
            if (directory / "cancel.request").exists():
                connection.put_json({}, remote_job / "cancel.request")
            if not launched:
                connection.command(["launch", directory.name]); launched = True
            state = connection.command(["snapshot", directory.name])
            state.update(execution="remote", remote_managed=True, remote_host=config.host, remote_pid=state.pop("pid", None),
                         synced_at=utc_now(), connection_status="connected", sync_error=None)
            try: state["remote_hardware"] = connection.get_json(remote_job / "hardware.json")
            except FileNotFoundError: pass
            try:
                with connection.sftp.open(str(remote_job / "worker.log"), "rb") as stream:
                    offset = (directory / "worker.log").stat().st_size if (directory / "worker.log").exists() else 0
                    stream.seek(offset); content = stream.read()
                with (directory / "worker.log").open("ab") as output: output.write(content)
            except FileNotFoundError: pass
            if state["status"] in TERMINAL:
                final = dict(state)
                state.update(status="running", phase="syncing", progress=0.995, eta_seconds=0)
                write_json(directory / "status.json", state)
                if final.get("run_dir"):
                    run = relative_file(final["run_dir"], "runs")
                    connection.tree(remote / run, paths.ROOT_DIR / str(run))
                if final["status"] == "completed" and request["kind"] == "training":
                    connection.tree(remote_job / "evaluation", directory / "evaluation")
                    detection = request["config"].get("backend") == "yolo"
                    filename = "best.pt" if detection else "best.pth"
                    archive = paths.TRAINED_MODELS_DIR / (("yolo-remote-" if detection else "mixercseg-remote-") + directory.name + "-" + filename)
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(paths.ROOT_DIR / final["run_dir"] / "weights" / filename, archive)
                    final["remote_best_weight"] = final.get("best_weight"); final["best_weight"] = str(archive)
                final["artifacts_synced_at"] = utc_now()
                write_json(directory / "status.json", final)
                break
            write_json(directory / "status.json", state)
        except (ValueError, RemoteRejected) as error:
            # 提交前校验失败可明确结束;已经启动的任务不能因 SSH 暂断被误判失败。
            if not launched:
                state.update(status="failed", phase="failed", error=str(error), finished_at=utc_now())
                write_json(directory / "status.json", state); break
            state.update(connection_status="disconnected", sync_error=str(error))
            write_json(directory / "status.json", state)
            if connection is not None: connection.close()
            connection = None
        except Exception as error:
            state.update(connection_status="disconnected", sync_error=str(error))
            write_json(directory / "status.json", state)
            if connection is not None: connection.close()
            connection = None
        time.sleep(4)
    if connection is not None: connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--job-dir", type=Path, required=True)
    bridge(parser.parse_args().job_dir)
