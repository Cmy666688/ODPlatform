"""Django 服务入口与共享的硬件、配置查询。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
import psutil

from od_platform.common import paths
from od_platform.runtime_config.mixercseg import MixerCSegConfig
from od_platform.runtime_config.detection import DetectionConfig

from .jobs import JobStore


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["training", "inference"]
    name: str = Field(default="WSODD 完整实验", min_length=1, max_length=80)
    config: Annotated[Union[MixerCSegConfig, DetectionConfig], Field(discriminator="backend")] = Field(default_factory=DetectionConfig)
    require_cuda: bool = True
    min_f1: Optional[float] = Field(default=None, ge=0, le=1)
    upload_id: Optional[str] = None
    model_id: str = "official"
    media_type: Literal["image", "video"] = "image"

    @model_validator(mode="before")
    @classmethod
    def legacy_backend(cls, value):
        if isinstance(value, dict) and isinstance(value.get("config"), dict):
            config = value["config"]
            if "backend" not in config:
                data = config.get("data", "wsodd")
                backend = "mixercseg" if data == "crackmap" else "yolo"
                value = {**value, "config": {**config, "backend": backend}}
        return value


def _hardware() -> dict:
    from .remote import hardware
    return hardware()


def _local_hardware() -> dict:
    import torch
    cuda = torch.cuda.is_available()
    memory = psutil.virtual_memory()
    result = {"cuda_available": cuda, "device": torch.cuda.get_device_name(0) if cuda else "CPU",
            "torch": torch.__version__, "cpu_percent": psutil.cpu_percent(),
            "memory_percent": memory.percent, "memory_gb": round(memory.total / 2**30, 1),
            "memory_used_gb": round(memory.used / 2**30, 1),
            "gpu_percent": None, "gpu_memory_used_gb": None, "gpu_memory_gb": None}
    if cuda:
        try:
            import pynvml
        except ImportError:
            pass
        else:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                result.update(gpu_percent=pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                              gpu_memory_used_gb=round(info.used / 2**30, 1), gpu_memory_gb=round(info.total / 2**30, 1))
            except pynvml.NVMLError:
                pass
            finally:
                try: pynvml.nvmlShutdown()
                except pynvml.NVMLError: pass
    return result


def _reference_dir() -> Path | None:
    for directory in sorted((paths.RUNS_DIR / "semantic_infer").glob("*"), reverse=True):
        audit_path = directory / "odp_audit.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if "checkpoint_CrackMap" in str(audit.get("config", {}).get("model", "")) and len(list((directory / "comparisons").glob("*.png"))) == 24:
                return directory
    return None


def create_app(store: JobStore | None = None):
    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "od_web.settings")
    from django.core.wsgi import get_wsgi_application
    application = get_wsgi_application()
    def handler(environ, start_response):
        if store is not None: environ["od.job_store"] = store
        return application(environ, start_response)
    return handler


def configure_environment() -> None:
    import os
    import secrets
    directory = paths.META_DIR / "web"; directory.mkdir(parents=True, exist_ok=True)
    secret_path = directory / "django-secret"
    if not secret_path.exists(): secret_path.write_text(secrets.token_urlsafe(48), encoding="utf-8")
    os.environ.setdefault("ODP_DJANGO_SECRET", secret_path.read_text(encoding="utf-8"))
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "od_web.settings")


def main() -> None:
    import django
    import uvicorn
    parser = argparse.ArgumentParser(); parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    configure_environment()
    django.setup()
    from django.core.management import call_command
    from django.core.asgi import get_asgi_application
    call_command("migrate", interactive=False, verbosity=0)
    call_command("collectstatic", interactive=False, verbosity=0)
    from od_platform.common.retention import start_scheduler
    start_scheduler()
    uvicorn.run(get_asgi_application(), host="127.0.0.1", port=arguments.port, lifespan="off")


if __name__ == "__main__": main()
