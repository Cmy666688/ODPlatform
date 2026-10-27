"""Django 页面与 JSON 接口,复用核心训练和已有实验现场。"""
from __future__ import annotations

from functools import wraps
import json
from pathlib import Path
from uuid import uuid4
from urllib.parse import quote

from django.http import FileResponse, HttpResponse, HttpResponseNotAllowed, JsonResponse
from django.http.response import HttpResponseBase
from django.shortcuts import render
from django.views.decorators.csrf import ensure_csrf_cookie
from pydantic import ValidationError

from od_platform.common import paths
from od_platform.runtime_config.mixercseg import MixerCSegConfig, load_mixercseg_config
from od_platform.runtime_config.detection import DetectionConfig

from . import app, assistant, datasets
from .forms import DatasetUploadForm
from .jobs import JobStore


class ApiError(Exception):
    def __init__(self, status, detail): self.status, self.detail = status, detail


def endpoint(methods):
    def decorate(function):
        @wraps(function)
        def view(request, *args, **kwargs):
            if request.method not in methods: return HttpResponseNotAllowed(methods)
            try:
                value = function(request, *args, **kwargs)
                return value if isinstance(value, HttpResponseBase) else JsonResponse(value, safe=False, json_dumps_params={"ensure_ascii": False})
            except ApiError as error: return JsonResponse({"detail": error.detail}, status=error.status)
            except ValidationError as error: return JsonResponse({"detail": error.errors(include_context=False)}, status=422)
            except FileNotFoundError as error: return JsonResponse({"detail": str(error)}, status=404)
            except (ValueError, OSError, KeyError, TypeError) as error: return JsonResponse({"detail": str(error)}, status=422)
            except RuntimeError as error: return JsonResponse({"detail": str(error)}, status=409)
        return view
    return decorate


def store(request): return request.META.get("od.job_store") or JobStore()


def csrf_failure(request, reason=""):
    return JsonResponse({"detail": "请求校验失败,请刷新页面后重试"}, status=403)


@ensure_csrf_cookie
def index(request): return render(request, "index.html")


@endpoint(["GET"])
def health(request): return {"status": "ok", "framework": "django", "hardware": app._hardware()}


@endpoint(["GET"])
def assistant_config(request): return assistant.status()


@endpoint(["POST"])
def assistant_chat(request):
    if len(request.body) > 128 * 1024: raise ApiError(413, "对话内容过大,请清空历史后重试")
    document = json.loads(request.body)
    if not isinstance(document, dict) or set(document) - {"message", "dataset", "job_id", "history"}:
        raise ApiError(422, "对话请求格式无效")
    evidence = assistant.context(datasets.describe(document.get("dataset", "wsodd")), store(request), document.get("job_id", ""))
    try: return assistant.ask(document.get("message", ""), evidence, document.get("history"))
    except assistant.ProviderError as error: raise ApiError(503, str(error)) from None


@endpoint(["GET", "POST"])
def explain_video(request, job_id):
    from . import explanation
    return explanation.start(store(request), job_id) if request.method == "POST" else explanation.read(store(request), job_id)


@endpoint(["GET", "POST"])
def retention_policy(request):
    from od_platform.common import retention
    if request.method == "POST": retention.configure(json.loads(request.body))
    return {**retention.preview(), "trash": retention.trash()}


@endpoint(["POST"])
def retention_apply(request):
    from od_platform.common.retention import sweep
    return sweep(force=True)


@endpoint(["POST"])
def retention_restore(request, identifier):
    from od_platform.common.retention import restore
    return restore(identifier)


@endpoint(["GET"])
def config(request):
    identifier = request.GET.get("dataset", "wsodd")
    semantic = datasets.describe(identifier)["task"] != "detect"
    defaults = load_mixercseg_config("mixercseg_full", {"device": "cuda", "scan_backend": "reference"}) if semantic else DetectionConfig()
    defaults.data = identifier
    return {"defaults": defaults.model_dump(), "schema": (MixerCSegConfig if semantic else DetectionConfig).model_json_schema(), "execution": "remote"}


@endpoint(["GET"])
def overview(request):
    choices = datasets.listing()
    selected = request.GET.get("dataset", "wsodd")
    dataset = next((row for row in choices if row["id"] == selected), None)
    if dataset is None: raise ApiError(404, "数据集不存在")
    return {"framework": "django", "hardware": app._hardware(), "dataset": dataset, "datasets": choices,
            "quality": dataset["quality"], "reference_available": app._reference_dir() is not None, "jobs": store(request).list()}


@endpoint(["GET", "POST"])
def dataset_list(request):
    if request.method == "GET": return datasets.listing()
    form = DatasetUploadForm(request.POST)
    if not form.is_valid(): raise ApiError(422, form.errors.get_json_data())
    return JsonResponse(datasets.receive(request, **form.cleaned_data), status=202)


@endpoint(["GET"])
def dataset_detail(request, identifier): return datasets.describe(identifier)


@endpoint(["GET", "POST"])
def validate(request, identifier):
    if request.method == "GET": return datasets.validation_status(identifier)
    if request.GET.get("background") == "true": return JsonResponse(datasets.launch_validation(identifier), status=202)
    return datasets.validate(identifier)


@endpoint(["GET"])
def quality_history(request, identifier):
    from od_platform.data_validation.history import listing
    datasets.dataset_yaml(identifier)
    return [{key: value.get(key) for key in ("report_id", "checked_at", "overall_severity", "comparison", "data_fingerprint")}
            for value in listing(identifier)]


@endpoint(["GET"])
def dataset_statistics(request, identifier):
    from collections import Counter, defaultdict
    import yaml
    selected = datasets.describe(identifier)
    if selected["task"] != "detect" or not selected["available"]: raise ApiError(409, "目标检测数据集尚未就绪")
    document = yaml.safe_load(datasets.dataset_yaml(identifier).read_text(encoding="utf-8"))
    manifest_file = (paths.ROOT_DIR / document["manifest_path"]).resolve()
    if not manifest_file.is_relative_to(paths.PROCESSED_DATA_DIR.resolve()): raise ApiError(422, "数据契约路径异常")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")); groups = defaultdict(Counter); sizes = Counter()
    for row in manifest["samples"]:
        groups[row.get("group", "未记录")][row["split"]] += 1
        sizes[f"{row['width']} × {row['height']}"] += 1
    quality = selected.get("quality") or {}
    return {"dataset": selected["name"], "id": identifier, "splits": selected["splits"], "names": manifest["names"],
            "class_counts": quality.get("class_counts", {}), "groups": dict(groups), "resolutions": dict(sizes),
            "source_format": manifest.get("source_format", "VOC"), "split_policy": manifest.get("split_policy", "未记录"),
            "fingerprint": selected["fingerprint"], "quality": quality.get("overall_severity"), "sample_count": len(manifest["samples"])}


@endpoint(["GET"])
def dataset_preview(request, identifier, split, filename, kind):
    if datasets.describe(identifier)["task"] == "detect":
        import io
        from od_platform.common.semantic import read_bgr
        from od_platform.training.detection_experiment import draw_boxes
        import yaml
        import cv2
        if split not in ("train", "val", "test") or Path(filename).name != filename or not filename.endswith(".jpg") or kind not in ("images", "masks"):
            raise ApiError(404, "预览不存在")
        document = yaml.safe_load(datasets.dataset_yaml(identifier).read_text(encoding="utf-8"))
        root = paths.ROOT_DIR / document["path"]
        image = root / "images" / split / filename
        if kind == "images": return FileResponse(image.open("rb"), content_type="image/jpeg")
        rows = [[float(v) for v in line.split()] for line in (root / "labels" / split / (Path(filename).stem + ".txt")).read_text().splitlines()]
        _, encoded = cv2.imencode(".jpg", draw_boxes(read_bgr(image), rows, document["names"]))
        return FileResponse(io.BytesIO(encoded.tobytes()), content_type="image/jpeg")
    if split not in ("train", "val", "test") or kind not in ("images", "masks"): raise ApiError(404, "预览不存在")
    from od_platform.common.semantic import load_semantic_dataset
    _, root = load_semantic_dataset(datasets.dataset_yaml(identifier))
    candidate = (root / kind / split / filename).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file() or candidate.suffix.lower() != ".png":
        raise ApiError(404, "预览不存在")
    return FileResponse(candidate.open("rb"), content_type="image/png")


@endpoint(["GET"])
def previews(request, identifier):
    if datasets.describe(identifier)["task"] == "detect":
        root = paths.PROCESSED_DATA_DIR / identifier
        return [{"name": path.name, "split": split, "image": f"/api/datasets/{identifier}/preview/images/{split}/{quote(path.name)}",
                 "mask": f"/api/datasets/{identifier}/preview/masks/{split}/{quote(path.name)}", "annotation_type": "boxes"}
                for split in ("train", "val", "test") for path in sorted((root / "images" / split).glob("*.jpg"))[:4]]
    from od_platform.common.semantic import load_semantic_dataset
    document, root = load_semantic_dataset(datasets.dataset_yaml(identifier))
    return [{"name": path.name, "split": split,
             "image": f"/api/datasets/{identifier}/preview/images/{split}/{quote(path.name)}",
             "mask": f"/api/datasets/{identifier}/preview/masks/{split}/{quote(path.name)}"}
            for split in ("train", "val", "test") for path in sorted((root / "images" / split).glob("*.png"))[:4]]


@endpoint(["GET", "POST"])
def jobs(request):
    if request.method == "GET": return store(request).list()
    submitted = app.JobRequest(**json.loads(request.body))
    configuration = submitted.config
    # 执行设备由服务端固定,旧草稿中的 CPU/auto 不会触发本机回退。
    configuration.device = "cuda"; configuration.scan_backend = "reference"
    submitted.require_cuda = True
    if submitted.kind == "training":
        if configuration.backend == "yolo" and submitted.min_f1 is not None:
            raise ApiError(422, "目标检测使用 mAP 与类别指标评审,不套用分割 F1 目标")
        if configuration.max_train_batches is not None or configuration.max_eval_batches is not None:
            raise ApiError(422, "完整实验不允许限批冒烟配置")
        if configuration.model is not None: raise ApiError(422, "完整实验从头训练,模型路径由服务端管理")
        selected = datasets.describe(configuration.data)
        if (configuration.backend == "yolo") != (selected.get("task") == "detect"):
            raise ApiError(422, "模型任务与数据标注不匹配,请刷新训练配置")
        if not selected["available"] or selected["status"] != "ready": raise ApiError(422, "数据集尚未就绪或存在阻断问题")
        report = datasets.validate(configuration.data, quick=True) if configuration.backend == "yolo" else datasets.validate(configuration.data)
        if report["overall_severity"] == "ERROR": raise ApiError(422, "数据质检 ERROR,请修复后训练")
        document = submitted.model_dump(exclude={"upload_id", "model_id"})
        document["dataset_name"] = selected["name"]
        document["dataset_counts"] = selected["splits"]
        document["dataset_fingerprint"] = selected["fingerprint"]
    else:
        import re
        from od_platform.common.video import VIDEO_SUFFIXES
        suffix = Path(submitted.upload_id or "").suffix
        allowed = VIDEO_SUFFIXES if submitted.media_type == "video" else (".png",)
        if not re.fullmatch(r"[0-9a-f]{32}\.[a-z0-9]{3}", submitted.upload_id or "") or suffix not in allowed:
            raise ApiError(422, "请先上传对应类型的图片或视频")
        upload_root = paths.RUNS_DIR / "web_uploads"; source = (upload_root / submitted.upload_id).resolve()
        if not source.is_relative_to(upload_root.resolve()) or not source.is_file(): raise ApiError(404, "上传图片不存在")
        data = "crackmap"
        if submitted.model_id == "official":
            checkpoint = paths.PRETRAINED_MODELS_DIR / "checkpoint_CrackMap.pth"; bins = 180
        elif submitted.model_id.startswith("trained:"):
            trained = store(request).read(submitted.model_id[8:])
            if trained["status"] != "completed" or trained.get("kind") != "training": raise ApiError(422, "训练任务未完成")
            checkpoint = Path(trained["best_weight"]); bins = trained["config"].get("hog_bins", 36); data = trained["config"]["data"]
        else: raise ApiError(422, "未知模型")
        if not checkpoint.is_file(): raise ApiError(409, "所选模型权重尚未就绪")
        if submitted.model_id.startswith("trained:") and trained["config"].get("backend") == "yolo":
            configuration = DetectionConfig(model=str(checkpoint), data=data, threshold=configuration.threshold,
                                            imgsz=trained["config"]["imgsz"])
        else:
            if submitted.media_type == "video": raise ApiError(422, "视频仅支持已训练的目标检测模型")
            configuration = load_mixercseg_config("mixercseg_infer", {"model": str(checkpoint), "hog_bins": bins,
                         "data": data, "threshold": configuration.threshold, "device": "cuda",
                         "scan_backend": "reference"})
        if submitted.media_type == "video":
            from od_platform.common.video import inspect_video
            inspect_video(source)
        document = {"kind": "inference", "name": submitted.name, "config": configuration.model_dump(), "source": str(source), "media_type": submitted.media_type}
    hardware = app._hardware()
    if not hardware["cuda_available"]:
        raise ApiError(409, hardware.get("message", "远程 GPU 尚未就绪,任务未启动"))
    return JsonResponse(store(request).create(document), status=202)


@endpoint(["GET"])
def job(request, job_id):
    try: return store(request).read(job_id)
    except ValueError as error: raise ApiError(404, str(error)) from error


@endpoint(["POST"])
def cancel(request, job_id):
    try: return store(request).cancel(job_id)
    except ValueError as error: raise ApiError(409, str(error)) from error


@endpoint(["POST"])
def upload(request):
    from od_platform.common.semantic import write_png
    import cv2
    import numpy as np
    file = request.FILES.get("file")
    if file is None: raise ApiError(422, "请选择图片")
    if file.size > 20 * 1024**2: raise ApiError(413, "图片不能超过 20 MB")
    decoded = cv2.imdecode(np.frombuffer(file.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size > 48_000_000: raise ApiError(422, "无法读取图片或图片尺寸过大")
    upload_id = uuid4().hex + ".png"; write_png(paths.RUNS_DIR / "web_uploads" / upload_id, decoded)
    return {"id": upload_id, "width": decoded.shape[1], "height": decoded.shape[0]}


@endpoint(["POST"])
def video_upload(request):
    from od_platform.common.video import MAX_VIDEO_BYTES, VIDEO_SUFFIXES, inspect_video
    file = request.FILES.get("file")
    if file is None: raise ApiError(422, "请选择视频")
    suffix = Path(file.name).suffix.lower()
    if suffix not in VIDEO_SUFFIXES or file.size > MAX_VIDEO_BYTES: raise ApiError(413, "请选择最大 200 MB 的 MP4、AVI、MOV 或 MKV 视频")
    identifier = uuid4().hex + suffix; target = paths.RUNS_DIR / "web_uploads" / identifier; target.parent.mkdir(parents=True, exist_ok=True)
    try:
        size = 0
        with target.open("xb") as stream:
            for chunk in file.chunks():
                size += len(chunk)
                if size > MAX_VIDEO_BYTES: raise ValueError("视频超过 200 MB")
                stream.write(chunk)
        return {"id": identifier, **inspect_video(target)}
    except Exception:
        target.unlink(missing_ok=True); raise


@endpoint(["GET"])
def artifacts(request, job_id):
    selected = store(request).read(job_id); entries = []; directory = store(request).directory(job_id)
    for root, prefix in [(directory / "evaluation", "evaluation/"), (paths.ROOT_DIR / selected.get("run_dir", "runs/nonexistent"), "run/")]:
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.suffix.lower() in (".json", ".jsonl", ".mp4", ".csv", ".png", ".jpg", ".md", ".pth", ".pt"):
                    relative = prefix + path.relative_to(root).as_posix()
                    entries.append({"path": relative, "name": path.name, "url": f"/api/jobs/{job_id}/files/{quote(relative, safe='/')}"})
    return entries


@endpoint(["GET"])
def artifact(request, job_id, relative):
    path = store(request).artifact(job_id, relative)
    if path.suffix == ".mp4":
        import asyncio
        import re
        from django.core.handlers.asgi import ASGIRequest
        from django.http import StreamingHttpResponse
        from django.utils.http import content_disposition_header
        size = path.stat().st_size; start = 0; end = size-1; range_header = request.META.get("HTTP_RANGE")
        if range_header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header)
            if not match or not any(match.groups()): return HttpResponse(status=416, headers={"Content-Range": f"bytes */{size}"})
            first, last = match.groups()
            start = int(first) if first else max(0, size-int(last)); end = min(size-1, int(last)) if first and last else size-1
            if start > end or start >= size: return HttpResponse(status=416, headers={"Content-Range": f"bytes */{size}"})
        def chunks():
            with path.open("rb") as stream:
                stream.seek(start); remaining = end-start+1
                while remaining:
                    data = stream.read(min(remaining, 1024**2))
                    if not data: break
                    remaining -= len(data); yield data
        async def async_chunks():
            stream = await asyncio.to_thread(path.open, "rb")
            try:
                await asyncio.to_thread(stream.seek, start); remaining = end-start+1
                while remaining > 0:
                    data = await asyncio.to_thread(stream.read, min(remaining, 1024**2))
                    if not data: break
                    remaining -= len(data); yield data
            finally: await asyncio.to_thread(stream.close)
        headers = {"Accept-Ranges": "bytes", "Content-Length": str(max(0, end-start+1))}
        if range_header: headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        if request.GET.get("download") == "true": headers["Content-Disposition"] = content_disposition_header(True, path.name)
        return StreamingHttpResponse(async_chunks() if isinstance(request, ASGIRequest) else chunks(),
            status=206 if range_header else 200, content_type="video/mp4", headers=headers)
    return FileResponse(path.open("rb"), as_attachment=request.GET.get("download") == "true", filename=path.name)


@endpoint(["GET"])
def reference(request):
    root = app._reference_dir()
    if root is None: return {"kind": "official_reference", "images": [], "run_id": None}
    return {"kind": "official_reference", "run_id": root.name,
            "images": [{"name": p.name, "url": "/api/reference/files/" + quote(p.name)} for p in sorted((root / "comparisons").glob("*.png"))]}


@endpoint(["GET"])
def reference_file(request, filename):
    root = app._reference_dir()
    if root is None or Path(filename).name != filename: raise ApiError(404, "图片不存在")
    path = root / "comparisons" / filename
    if not path.is_file() or path.suffix != ".png": raise ApiError(404, "图片不存在")
    return FileResponse(path.open("rb"), content_type="image/png")


@endpoint(["GET"])
def dataset_template(request):
    import io
    import zipfile
    import cv2
    import numpy as np
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("demo/README.txt", "Synthetic harbor detection samples for workflow testing only. Each label row: class cx cy width height (normalized). Replace with real images/labels.\n")
        archive.writestr("demo/data.yaml", "names: [boat, ship, buoy]\n")
        for index in range(9):
            rng = np.random.default_rng(index)
            image = rng.integers(150, 220, (64, 64, 3), dtype=np.uint8)
            for x, color in [(4, (60, 90, 160)), (24, (70, 150, 60)), (44, (200, 180, 40))]:
                cv2.rectangle(image, (x, 20), (x+12, 44), color, -1)
            success, encoded = cv2.imencode(".png", image)
            if not success: raise ValueError("模板生成失败")
            stem = f"harbor{index:02d}_0001"
            archive.writestr(f"demo/images/{stem}.png", encoded.tobytes())
            archive.writestr(f"demo/labels/{stem}.txt", "0 0.15625 0.5 0.1875 0.375\n1 0.46875 0.5 0.1875 0.375\n2 0.78125 0.5 0.1875 0.375\n")
    response = HttpResponse(buffer.getvalue(), content_type="application/zip")
    response["Content-Disposition"] = 'attachment; filename="ODPlatform-dataset-template.zip"'
    return response
