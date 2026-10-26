"""大包上传入口边界,文件仍由 Django 临时文件处理器分块落盘。"""
from django.http import JsonResponse

from od_platform.data_pipeline.detection_import import MAX_UPLOAD


class DatasetUploadLimit:
    def __init__(self, get_response): self.get_response = get_response

    def __call__(self, request):
        if request.method == "POST" and request.path == "/api/datasets":
            try: length = int(request.META.get("CONTENT_LENGTH") or 0)
            except ValueError: length = 0
            if length > MAX_UPLOAD + 32 * 1024**2:
                return JsonResponse({"detail": "单次上传总大小不能超过 4 GB"}, status=413)
        return self.get_response(request)
