from django.contrib import admin
from django.urls import path
from django.views.static import serve

from . import views
from .settings import BASE_DIR, STATIC_ROOT

urlpatterns = [
    path("", views.index), path("admin/", admin.site.urls),
    path("api/health", views.health), path("api/config", views.config), path("api/overview", views.overview),
    path("api/assistant/config", views.assistant_config), path("api/assistant/chat", views.assistant_chat),
    path("api/datasets", views.dataset_list), path("api/datasets/template", views.dataset_template),
    path("api/datasets/<str:identifier>", views.dataset_detail),
    path("api/datasets/<str:identifier>/history", views.quality_history),
    path("api/datasets/<str:identifier>/validate", views.validate), path("api/datasets/<str:identifier>/previews", views.previews),
    path("api/datasets/<str:identifier>/preview/<str:kind>/<str:split>/<str:filename>", views.dataset_preview),
    path("api/jobs", views.jobs), path("api/jobs/<str:job_id>", views.job),
    path("api/jobs/<str:job_id>/cancel", views.cancel), path("api/jobs/<str:job_id>/artifacts", views.artifacts),
    path("api/jobs/<str:job_id>/files/<path:relative>", views.artifact), path("api/uploads", views.upload),
    path("api/reference", views.reference), path("api/reference/files/<str:filename>", views.reference_file),
    path("assets/admin/<path:path>", serve, {"document_root": STATIC_ROOT / "admin"}),
    path("assets/<path:path>", serve, {"document_root": BASE_DIR / "public"}),
]
