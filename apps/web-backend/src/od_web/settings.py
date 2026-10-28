"""Django 本机工作台配置,数据库与上传现场均存于工作区。"""
import os
from pathlib import Path
import secrets

from od_platform.common import paths

BASE_DIR = Path(__file__).resolve().parent
SECRET_KEY = os.environ.get("ODP_DJANGO_SECRET", secrets.token_urlsafe(48))
DEBUG = False
# 公网部署时通过环境变量追加主机名/IP 与可信源(逗号分隔),本机开发默认不变。
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"] + [h.strip() for h in os.environ.get("ODP_ALLOWED_HOSTS", "").split(",") if h.strip()]
CSRF_TRUSTED_ORIGINS = [o.strip() for o in os.environ.get("ODP_CSRF_ORIGINS", "").split(",") if o.strip()]
# 反向代理/隧道终止 TLS 后,浏览器地址栏是 https 而 uvicorn 收到的是明文 http。
# 不告知 Django 真实协议,CSRF 校验会因 scheme 不一致把合法的 https 表单判为非法。
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
ROOT_URLCONF = "od_web.urls"
INSTALLED_APPS = ["django.contrib.admin", "django.contrib.auth", "django.contrib.contenttypes",
                  "django.contrib.sessions", "django.contrib.messages", "django.contrib.staticfiles", "od_web"]
MIDDLEWARE = ["django.middleware.security.SecurityMiddleware", "django.contrib.sessions.middleware.SessionMiddleware",
              "django.middleware.common.CommonMiddleware", "od_web.upload_limits.DatasetUploadLimit", "django.middleware.csrf.CsrfViewMiddleware",
              "django.contrib.auth.middleware.AuthenticationMiddleware", "django.contrib.messages.middleware.MessageMiddleware"]
TEMPLATES = [{"BACKEND": "django.template.backends.django.DjangoTemplates", "DIRS": [BASE_DIR / "public"],
              "APP_DIRS": True, "OPTIONS": {"context_processors": ["django.template.context_processors.request",
              "django.contrib.auth.context_processors.auth", "django.contrib.messages.context_processors.messages"]}}]
DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": paths.META_DIR / "web/platform.sqlite3",
                         "OPTIONS": {"timeout": 30}}}
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
STATIC_URL = "/assets/"
STATIC_ROOT = paths.META_DIR / "web/static"
STATICFILES_DIRS = [BASE_DIR / "public"]
MEDIA_ROOT = paths.META_DIR / "dataset_uploads"
FILE_UPLOAD_MAX_MEMORY_SIZE = 2 * 1024**2
DATA_UPLOAD_MAX_MEMORY_SIZE = 32 * 1024**2
DATA_UPLOAD_MAX_NUMBER_FILES = 100000
DATA_UPLOAD_MAX_NUMBER_FIELDS = 100
CSRF_FAILURE_VIEW = "od_web.views.csrf_failure"
USE_TZ = True
TIME_ZONE = "Asia/Shanghai"
LANGUAGE_CODE = "zh-hans"
APPEND_SLASH = False
X_FRAME_OPTIONS = "DENY"
