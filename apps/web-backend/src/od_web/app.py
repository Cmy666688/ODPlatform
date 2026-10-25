"""FastAPI API 与正式 Web 面板,核心引擎不依赖前端框架。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Annotated, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
import psutil

from od_platform.common import paths, refs
from od_platform.common.semantic import load_semantic_dataset
from od_platform.runtime_config.mixercseg import MixerCSegConfig, load_mixercseg_config

from .jobs import JobStore


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["training", "inference"]
    name: str = Field(default="CrackMap 完整实验", min_length=1, max_length=80)
    config: MixerCSegConfig = Field(default_factory=MixerCSegConfig)
    require_cuda: bool = True
    min_f1: Optional[float] = Field(default=None, ge=0, le=1)
    upload_id: Optional[str] = None
    model_id: str = "official"


def _hardware() -> dict:
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


def create_app(store: JobStore | None = None) -> FastAPI:
    app = FastAPI(title="ODPlatform API", version="0.1.0")
    app.state.store = store or JobStore()
    public = Path(__file__).parent / "public"
    app.mount("/assets", StaticFiles(directory=public), name="assets")

    @app.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method in ("POST", "PUT", "DELETE") and origin and origin != str(request.base_url).rstrip("/"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "跨来源写入请求被拒绝"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/")
    def index(): return FileResponse(public / "index.html")

    @app.get("/api/health")
    def health(): return {"status": "ok", "hardware": _hardware()}

    @app.get("/api/config")
    def config():
        return {"defaults": load_mixercseg_config("mixercseg_full").model_dump(),
                "schema": MixerCSegConfig.model_json_schema()}

    @app.get("/api/overview")
    def overview():
        yaml_path = refs.resolve_dataset_yaml("crackmap")
        dataset = {"name": "CrackMap", "available": False, "splits": {"train": 0, "val": 0, "test": 0}}
        if yaml_path.exists():
            document, root = load_semantic_dataset(yaml_path)
            dataset.update(available=root.exists(), fingerprint=document["data_fingerprint"],
                           splits={split: len(list((root / "images" / split).glob("*.png"))) for split in ("train", "val", "test")})
        quality_path = paths.RUNS_DIR / "data_validation/crackmap/report.json"
        quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else None
        reference = _reference_dir()
        return {"hardware": _hardware(), "dataset": dataset, "quality": quality,
                "reference_available": reference is not None, "jobs": app.state.store.list()}

    @app.post("/api/datasets/crackmap/validate")
    def validate():
        from od_platform.data_validation.semantic import validate_semantic_dataset
        try:
            return validate_semantic_dataset(refs.resolve_dataset_yaml("crackmap"), run_id="crackmap").to_dict()
        except (OSError, ValueError) as e:
            raise HTTPException(422, str(e)) from e

    @app.get("/api/jobs")
    def jobs(): return app.state.store.list()

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        try: return app.state.store.read(job_id)
        except (ValueError, FileNotFoundError) as e: raise HTTPException(404, str(e)) from e

    @app.post("/api/jobs", status_code=202)
    def start(request: JobRequest):
        config = request.config
        if request.kind == "training":
            if config.max_train_batches is not None or config.max_eval_batches is not None:
                raise HTTPException(422, "完整实验不允许限批冒烟配置")
            if request.require_cuda and not _hardware()["cuda_available"]:
                raise HTTPException(409, "CUDA GPU 尚未就绪,完整训练未启动")
            if config.model is not None:
                raise HTTPException(422, "完整实验从头训练,模型路径由服务端管理")
            if config.data != "crackmap": raise HTTPException(422, "当前工作台支持 CrackMap")
            document = request.model_dump(exclude={"upload_id", "model_id"})
        else:
            if request.upload_id is None or len(request.upload_id) != 36 or not request.upload_id.endswith(".png"):
                raise HTTPException(422, "请先上传图片")
            upload_root = paths.RUNS_DIR / "web_uploads"
            source = (upload_root / request.upload_id).resolve()
            if not source.is_relative_to(upload_root.resolve()) or not source.is_file():
                raise HTTPException(404, "上传图片不存在")
            if request.model_id == "official":
                checkpoint = paths.PRETRAINED_MODELS_DIR / "checkpoint_CrackMap.pth"; bins = 180
            elif request.model_id.startswith("trained:"):
                try:
                    trained = app.state.store.read(request.model_id[8:])
                    if trained["status"] != "completed" or trained.get("kind") != "training": raise ValueError("训练任务未完成")
                    checkpoint = Path(trained["best_weight"]); bins = trained["config"]["hog_bins"]
                except (ValueError, KeyError, FileNotFoundError) as e:
                    raise HTTPException(422, str(e)) from e
            else: raise HTTPException(422, "未知模型")
            if not checkpoint.is_file(): raise HTTPException(409, "所选模型权重尚未就绪")
            hardware = _hardware()
            config = load_mixercseg_config("mixercseg_infer", {"model": str(checkpoint), "hog_bins": bins,
                        "threshold": config.threshold, "device": "auto",
                        "scan_backend": "cpu_jit" if not hardware["cuda_available"] and importlib.util.find_spec("numba") else "reference"})
            document = {"kind": "inference", "name": request.name, "config": config.model_dump(), "source": str(source)}
        try: return app.state.store.create(document)
        except RuntimeError as e: raise HTTPException(409, str(e)) from e

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        try: return app.state.store.cancel(job_id)
        except (ValueError, FileNotFoundError) as e: raise HTTPException(409, str(e)) from e

    @app.post("/api/uploads")
    async def upload(file: Annotated[UploadFile, File()]):
        body = await file.read(20 * 1024 * 1024 + 1)
        if len(body) > 20 * 1024 * 1024: raise HTTPException(413, "图片不能超过 20 MB")
        from od_platform.common.semantic import write_png
        import cv2
        import numpy as np
        try:
            decoded = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None or decoded.size > 48_000_000: raise ValueError("图片无效或尺寸过大")
            upload_id = uuid4().hex + ".png"
            write_png(paths.RUNS_DIR / "web_uploads" / upload_id, decoded)
            return {"id": upload_id, "width": decoded.shape[1], "height": decoded.shape[0]}
        except (ValueError, cv2.error) as e: raise HTTPException(422, "无法读取上传图片") from e

    @app.get("/api/jobs/{job_id}/artifacts")
    def artifacts(job_id: str):
        state = job(job_id)
        entries = []
        directory = app.state.store.directory(job_id)
        for root, prefix in [(directory / "evaluation", "evaluation/"),
                             (paths.ROOT_DIR / state.get("run_dir", "runs/nonexistent"), "run/")]:
            if root.exists():
                for path in sorted(root.rglob("*")):
                    if path.is_file() and path.suffix.lower() in (".json", ".csv", ".png", ".jpg", ".md", ".pth"):
                        relative = prefix + path.relative_to(root).as_posix()
                        entries.append({"path": relative, "name": path.name,
                                        "url": f"/api/jobs/{job_id}/files/{relative}"})
        return entries

    @app.get("/api/jobs/{job_id}/files/{relative:path}")
    def artifact(job_id: str, relative: str, download: bool = False):
        try:
            path = app.state.store.artifact(job_id, relative)
            return FileResponse(path, filename=path.name if download else None)
        except (ValueError, FileNotFoundError) as e: raise HTTPException(404, str(e)) from e

    @app.get("/api/reference")
    def reference():
        root = _reference_dir()
        if root is None: return {"kind": "official_reference", "images": [], "run_id": None}
        return {"kind": "official_reference", "run_id": root.name,
                "images": [{"name": p.name, "url": "/api/reference/files/" + p.name} for p in sorted((root / "comparisons").glob("*.png"))]}

    @app.get("/api/reference/files/{filename}")
    def reference_file(filename: str):
        root = _reference_dir()
        if root is None or Path(filename).name != filename: raise HTTPException(404, "图片不存在")
        path = root / "comparisons" / filename
        if not path.is_file() or path.suffix != ".png": raise HTTPException(404, "图片不存在")
        return FileResponse(path)

    return app


def main() -> None:
    import uvicorn
    parser = argparse.ArgumentParser(); parser.add_argument("--port", type=int, default=8000)
    arguments = parser.parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=arguments.port)


if __name__ == "__main__": main()
