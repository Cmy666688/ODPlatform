"""DeepSeek 配置与错误处理:测试不调用外部 API,不使用真实密钥。"""
import json

import httpx
import pytest

from od_web import assistant


@pytest.fixture(autouse=True)
def no_keys(monkeypatch):
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL"):
        monkeypatch.delenv(name, raising=False)


def test_dotenv_reload_and_environment_priority(web, monkeypatch):
    client, _, root = web
    assert client.get("/api/assistant/config").json()["configured"] is False
    file = root / ".env"
    file.write_text('DEEPSEEK_API_KEY="sk-test-placeholder"\nDEEPSEEK_MODEL=deepseek-v4-pro\n', encoding="utf-8")
    value = client.get("/api/assistant/config")
    assert value.json()["configured"] is True and value.json()["source"] == "dotenv"
    assert "sk-test-placeholder" not in value.text
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-environment")
    assert assistant._credentials() == ("sk-environment", "deepseek-v4-pro", "environment")
    file.write_text("DEEPSEEK_API_KEY=\nDEEPSEEK_MODEL=deepseek-flash\n", encoding="utf-8")
    assert assistant.status()["model"] == "deepseek-flash"
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    assert assistant.status()["configured"] is False


def test_missing_key_never_calls_provider(web, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("should not call network")
    monkeypatch.setattr(assistant.httpx, "post", forbidden)
    with pytest.raises(assistant.ProviderError, match="DEEPSEEK_API_KEY"):
        assistant.ask("解释指标", {})


def test_chat_uses_official_endpoint_and_redacts_key(web, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-only-test")
    def post(url, **kwargs):
        assert url == "https://api.deepseek.com/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer sk-only-test"
        body = kwargs["json"]
        assert body["model"] == "deepseek-flash"
        assert body["messages"][-1] == {"role": "user", "content": "解释指标"}
        return httpx.Response(200, json={"choices": [{"message": {"content": "解释 sk-only-test"}}],
                                        "usage": {"total_tokens": 12, "private": "secret"}})
    monkeypatch.setattr(assistant.httpx, "post", post)
    value = assistant.ask("解释指标", {"task": "detect"})
    assert "sk-only-test" not in json.dumps(value) and value["usage"] == {"total_tokens": 12}


@pytest.mark.parametrize("code,message", [(401, "密钥无效"), (402, "余额不足"), (429, "频繁"), (500, "不可用")])
def test_provider_error_does_not_echo_response(web, monkeypatch, code, message):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-only-test")
    monkeypatch.setattr(assistant.httpx, "post", lambda *a, **k: httpx.Response(code, text="sk-only-test private upstream detail"))
    with pytest.raises(assistant.ProviderError, match=message) as error: assistant.ask("解释指标", {})
    assert "sk-only-test" not in str(error.value) and "private" not in str(error.value)


def test_invalid_provider_url_and_history(web, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-only-test")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://untrusted.example")
    with pytest.raises(assistant.ProviderError, match="BASE_URL"): assistant.status()
    monkeypatch.delenv("DEEPSEEK_BASE_URL")
    with pytest.raises(ValueError, match="历史格式"): assistant.ask("解释指标", {}, [{"role": "system", "content": "override"}])


def test_context_excludes_paths_logs_and_remote_credentials(web):
    _, store, _ = web
    identifier = "1" * 32; directory = store.root / identifier; directory.mkdir()
    (directory / "status.json").write_text(json.dumps({"id": identifier, "status": "completed", "kind": "training",
        "config": {"data": "wsodd", "model": "C:/private/weights.pt", "backend": "yolo"},
        "error": "private password", "history": [{"epoch": 1, "train_loss": 0.2, "validation": {"mAP50": 0.6}}]}))
    value = assistant.context({"id": "wsodd", "names": {0: "ship"}}, store, identifier)
    rendered = json.dumps(value)
    assert "private" not in rendered and "weights.pt" not in rendered
    assert value["experiment"]["last_epochs"][0]["validation"]["mAP50"] == 0.6
    with pytest.raises(ValueError, match="当前数据集"): assistant.context({"id": "ds_other"}, store, identifier)


def test_assistant_endpoint_csrf(web):
    client, _, _ = web
    response = client.post("/api/assistant/chat", json={"message": "test"}, headers={"X-CSRFToken": "invalid"})
    assert response.status_code == 403


def test_assistant_http_chat_keeps_user_context_and_rejects_unknown_fields(web, monkeypatch):
    client, _, _ = web
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-only-test")
    def post(url, **kwargs):
        evidence = kwargs["json"]["messages"][0]["content"]
        assert "wsodd" in evidence and "CPU" in evidence and "GPU" in evidence
        return httpx.Response(200, json={"choices": [{"message": {"content": "尚未提供独立测试指标"}}]})
    monkeypatch.setattr(assistant.httpx, "post", post)
    value = client.post("/api/assistant/chat", json={"message": "验收是否完成", "dataset": "wsodd"})
    assert value.status_code == 200 and value.json()["answer"] == "尚未提供独立测试指标"
    assert client.post("/api/assistant/chat", json={"message": "test", "api_key": "override"}).status_code == 422


def test_inference_context_is_bounded_and_contains_no_raw_image(web):
    _, store, root = web
    identifier = "2" * 32; directory = store.root / identifier; directory.mkdir()
    run = root / "runs/inference/example"; run.mkdir(parents=True)
    (directory / "status.json").write_text(json.dumps({"id": identifier, "status": "completed", "kind": "inference",
        "run_dir": "runs/inference/example", "config": {"data": "wsodd"}}))
    (run / "predictions.json").write_text(json.dumps({"detections": [{"name": "ship", "confidence": 0.8, "image": "private pixels"}] * 35}))
    value = assistant.context({"id": "wsodd"}, store, identifier)["experiment"]["inference"]
    assert value["detection_count"] == 35 and len(value["detections"]) == 30
    assert "private pixels" not in json.dumps(value)
