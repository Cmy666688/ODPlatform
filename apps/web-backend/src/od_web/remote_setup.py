"""配置固定的 SSH GPU;口令仅用于安装专用密钥,不写入配置或日志。"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import getpass
import json
from pathlib import Path
import shlex
import tarfile

from od_platform.common import paths
from od_platform.common.json_io import write_json

from .remote import Connection, RemoteConfig, hardware


def source_bundle(target: Path):
    roots = [paths.ROOT_DIR / "apps/platform/src", paths.ROOT_DIR / "apps/web-backend/src",
             paths.ROOT_DIR / "apps/platform/configs"]
    with tarfile.open(target, "w:gz") as archive:
        archive.add(paths.ROOT_DIR / ".odp-workspace", arcname=".odp-workspace")
        for root in roots:
            for source in sorted(root.rglob("*")):
                if source.is_file() and not source.is_symlink() and "__pycache__" not in source.parts and source.suffix not in (".pyc", ".pth"):
                    archive.add(source, arcname=source.relative_to(paths.ROOT_DIR).as_posix())


def deploy(config: RemoteConfig):
    from pathlib import PurePosixPath
    directory = paths.META_DIR / "web"; directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "remote-source.tar.gz"; source_bundle(archive)
    connection = Connection(config)
    try:
        connection.mkdir(PurePosixPath(config.root))
        connection.put(archive, PurePosixPath(config.root) / "source.tar.gz")
        # 包只包含服务端生成的源代码,不提取用户上传的压缩包。
        code = "import tarfile; t=tarfile.open('source.tar.gz'); t.extractall('.'); t.close()"
        command = f"cd {shlex.quote(config.root)} && {shlex.quote(config.python)} -c {shlex.quote(code)}"
        _, stdout, stderr = connection.client.exec_command(command, timeout=60)
        if stdout.channel.recv_exit_status(): raise RuntimeError(stderr.read().decode())
        connection.command(["check"])
    finally: connection.close()


def main():
    import paramiko
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host"); parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--username", default="root"); parser.add_argument("--root")
    parser.add_argument("--python"); parser.add_argument("--extra-pythonpath", default="")
    parser.add_argument("--known-hosts"); parser.add_argument("--deploy-only", action="store_true")
    args = parser.parse_args()
    if args.deploy_only:
        config = RemoteConfig.load()
    else:
        if not all([args.host, args.root, args.python, args.known_hosts]): parser.error("必须提供 host/root/python/known-hosts")
        directory = paths.META_DIR / "web"; directory.mkdir(parents=True, exist_ok=True)
        key_path = directory / "remote_gpu_key"
        if not key_path.exists():
            key = paramiko.RSAKey.generate(3072); key.write_private_key_file(str(key_path)); key_path.chmod(0o600)
        else: key = paramiko.RSAKey.from_private_key_file(str(key_path))
        config = RemoteConfig(args.host, args.port, args.username, args.root, args.python,
                              str(key_path), str(Path(args.known_hosts).resolve()), args.extra_pythonpath)
        client = paramiko.SSHClient(); client.load_host_keys(config.known_hosts)
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        password = getpass.getpass("SSH password: ")
        try:
            client.connect(config.host, port=config.port, username=config.username, password=password,
                           timeout=15, look_for_keys=False, allow_agent=False)
            public = key.get_name() + " " + key.get_base64() + " ODPlatform-GPU\n"
            code = ("from pathlib import Path; import sys; d=Path.home()/'.ssh'; d.mkdir(mode=448,exist_ok=True); "
                    "p=d/'authorized_keys'; old=p.read_text() if p.exists() else ''; key=sys.stdin.read(); "
                    "p.write_text(old+('' if old.endswith('\\n') or not old else '\\n')+key) if key.strip() not in old else None; p.chmod(384)")
            stdin, stdout, stderr = client.exec_command(shlex.quote(config.python) + " -c " + shlex.quote(code))
            stdin.write(public); stdin.flush(); stdin.channel.shutdown_write()
            if stdout.channel.recv_exit_status(): raise RuntimeError(stderr.read().decode())
        finally:
            password = None; client.close()
        write_json(directory / "remote.json", asdict(config))
    deploy(config)
    print(json.dumps(hardware(force=True), ensure_ascii=False))


if __name__ == "__main__": main()
