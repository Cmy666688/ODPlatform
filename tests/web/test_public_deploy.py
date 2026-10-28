"""公网隧道部署配置测试。

覆盖:代理协议头已声明、隧道后请求被识别为 HTTPS、云隧道域名通过 CSRF 校验、
外部恶意来源仍被 CSRF 拦截。这些是 Cloudflare 隧道 + Django 组合最容易出问题的点。
"""
from django.test import RequestFactory


def test_proxy_ssl_header_declared(settings):
    assert settings.SECURE_PROXY_SSL_HEADER == ("HTTP_X_FORWARDED_PROTO", "https")


def test_request_is_secure_behind_tunnel():
    request = RequestFactory().get("/", HTTP_X_FORWARDED_PROTO="https")
    assert request.is_secure() is True


def test_request_stays_plain_without_proxy_header():
    request = RequestFactory().get("/")
    assert request.is_secure() is False


def test_tunnel_origin_passes_csrf(web, settings):
    client, _, _ = web
    settings.ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver", ".trycloudflare.com"]
    settings.CSRF_TRUSTED_ORIGINS = ["https://*.trycloudflare.com"]
    response = client.post(
        "/api/uploads",
        headers={"Host": "quiet-lab-1234.trycloudflare.com",
                 "Origin": "https://quiet-lab-1234.trycloudflare.com",
                 "X-Forwarded-Proto": "https"},
    )
    # 没有携带文件时业务层返回 422;关键是它没有在 CSRF 层被拦成 403。
    assert response.status_code == 422
    assert response.json()["detail"] == "请选择图片"


def test_foreign_origin_blocked(web, settings):
    client, _, _ = web
    settings.ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver", ".trycloudflare.com"]
    settings.CSRF_TRUSTED_ORIGINS = ["https://*.trycloudflare.com"]
    response = client.post(
        "/api/uploads",
        headers={"Host": "quiet-lab-1234.trycloudflare.com",
                 "Origin": "https://evil.example",
                 "X-Forwarded-Proto": "https"},
    )
    assert response.status_code == 403
    assert "请求校验失败" in response.json()["detail"]
