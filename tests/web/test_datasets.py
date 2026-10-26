"""用户数据集从上传、质检到训练请求的真实边界检查。"""
import io
import json
from pathlib import Path
import stat
import tarfile
import zipfile

import pytest

from od_platform.common import paths
from od_platform.common.semantic import load_semantic_dataset
from od_platform.data_pipeline import import_dataset
from od_web import datasets
from od_web.dataset_worker import run
from od_web.models import Dataset


@pytest.fixture
def inline_import(monkeypatch):
    monkeypatch.setattr(datasets, "launch_import", lambda record, source, mode: run(record.id, source, mode))


def upload_template(client, mode="archive"):
    blob = client.get("/api/datasets/template").content
    if mode == "folder":
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            entries = [(name, archive.read(name)) for name in archive.namelist() if name.endswith(".png")]
        return client.post("/api/datasets", data={"name": "道路样本", "mode": "folder", "relative_paths": json.dumps([name for name, _ in entries])},
                           files=[("files", (Path(name).name, content, "image/png")) for name, content in entries])
    if mode == "tar.gz":
        output = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(blob)) as archive, tarfile.open(fileobj=output, mode="w:gz") as target:
            for name in archive.namelist():
                data = archive.read(name); member = tarfile.TarInfo(name); member.size = len(data)
                target.addfile(member, io.BytesIO(data))
        blob = output.getvalue()
    return client.post("/api/datasets", data={"name": "道路样本", "mode": "archive"},
                       files={"files": ("dataset." + ("tar.gz" if mode == "tar.gz" else "zip"), blob)})


@pytest.mark.parametrize("mode", ["archive", "tar.gz", "folder"])
def test_dataset_import_materializes_pairs_quality_and_previews(web, inline_import, mode):
    client, _, _ = web
    response = upload_template(client, mode)
    assert response.status_code == 202, response.text
    result = response.json()
    assert result["status"] == "ready" and result["trainable"]
    assert result["splits"] == {"train": 7, "val": 1, "test": 1}
    identifier = result["id"]
    document, _ = load_semantic_dataset(paths.dataset_yaml_path(identifier))
    assert document["source"]["split_strategy"] == "grouped_seed42"
    assert client.get("/api/overview", params={"dataset": identifier}).json()["dataset"]["name"] == "道路样本"
    previews = client.get(f"/api/datasets/{identifier}/previews").json()
    assert previews and client.get(previews[0]["image"]).status_code == 200
    assert client.get(previews[0]["mask"]).status_code == 200
    assert client.post(f"/api/datasets/{identifier}/validate").json()["overall_severity"] != "ERROR"


def test_new_dataset_training_is_not_restricted_to_crackmap(web, inline_import, monkeypatch):
    client, store, _ = web
    identifier = upload_template(client).json()["id"]
    submitted = []
    monkeypatch.setattr(store, "create", lambda request: submitted.append(request) or {"id": "a" * 32})
    response = client.post("/api/jobs", json={"kind": "training", "require_cuda": False,
                           "config": {"data": identifier, "device": "cpu", "scan_backend": "cpu_jit", "epochs": 1}})
    assert response.status_code == 202, response.text
    assert submitted[0]["config"]["data"] == identifier
    assert submitted[0]["dataset_name"] == "道路样本" and submitted[0]["dataset_fingerprint"]
    assert client.post("/api/jobs", json={"kind": "training", "require_cuda": False,
                       "config": {"data": "../../secret"}}).status_code == 422


def test_failed_quality_prevents_training_after_dataset_is_modified(web, inline_import, monkeypatch):
    client, store, _ = web
    identifier = upload_template(client).json()["id"]
    next((paths.PROCESSED_DATA_DIR / identifier / "masks/train").glob("*.png")).unlink()
    def forbidden(*args, **kwargs): raise AssertionError("错误数据不允许训练")
    monkeypatch.setattr(store, "create", forbidden)
    response = client.post("/api/jobs", json={"kind": "training", "require_cuda": False, "config": {"data": identifier}})
    assert response.status_code == 422
    assert Dataset.objects.get(pk=identifier).status == "blocked"


@pytest.mark.parametrize("filename", ["../outside.png", "C:/outside.png", "folder/../../outside.png"])
def test_archive_traversal_is_rejected_before_writing_outside(web, inline_import, filename):
    client, _, root = web
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive: archive.writestr(filename, b"bad")
    response = client.post("/api/datasets", data={"name": "bad", "mode": "archive"}, files={"files": ("bad.zip", buffer.getvalue())})
    assert response.status_code == 422
    assert not (root / "outside.png").exists()
    assert Dataset.objects.get().status == "failed"


def test_upload_requires_csrf_token_even_without_login(web):
    client, _, _ = web
    del client.headers["X-CSRFToken"]
    assert client.post("/api/datasets", data={"name": "bad", "mode": "archive"}).status_code == 403


def test_provided_split_is_preserved_instead_of_randomly_resplit(web, inline_import):
    client, _, _ = web
    original = client.get("/api/datasets/template").content; output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(output, "w") as target:
        for index in range(9):
            split = "train" if index < 5 else "val" if index < 7 else "test"
            for kind in ("images", "masks"):
                target.writestr(f"custom/{kind}/{split}/sample_{index:02d}.png", source.read(f"demo/{kind}/sample_{index:02d}.png"))
    result = client.post("/api/datasets", data={"name": "已有划分", "mode": "archive"}, files={"files": ("split.zip", output.getvalue())})
    assert result.status_code == 202, result.text
    assert result.json()["splits"] == {"train": 5, "val": 2, "test": 2}
    document, _ = load_semantic_dataset(paths.dataset_yaml_path(result.json()["id"]))
    assert document["source"]["split_strategy"] == "provided"


@pytest.mark.parametrize("kind", ["zip_link", "tar_link", "too_large"])
def test_archives_reject_links_and_expansion_limits(tmp_path, monkeypatch, kind):
    archive_path = tmp_path / "input.archive"; buffer = io.BytesIO()
    if kind == "tar_link":
        with tarfile.open(fileobj=buffer, mode="w:gz") as target:
            member = tarfile.TarInfo("link"); member.type = tarfile.SYMTYPE; member.linkname = "../outside"
            target.addfile(member)
    else:
        with zipfile.ZipFile(buffer, "w") as target:
            if kind == "zip_link":
                member = zipfile.ZipInfo("link"); member.create_system = 3; member.external_attr = (stat.S_IFLNK | 0o777) << 16
                target.writestr(member, "../outside")
            else:
                monkeypatch.setattr(import_dataset, "MAX_FILE", 8)
                target.writestr("big.bin", b"x" * 9)
    archive_path.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError): import_dataset.extract_archive(archive_path, tmp_path / "extracted")
    assert not (tmp_path / "outside").exists()


def test_missing_annotation_is_an_import_error_not_a_trainable_dataset(web, inline_import):
    client, _, _ = web
    original = client.get("/api/datasets/template").content; output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(output, "w") as target:
        for name in source.namelist():
            if name != "demo/masks/sample_00.png": target.writestr(name, source.read(name))
    result = client.post("/api/datasets", data={"name": "缺失标注", "mode": "archive"}, files={"files": ("bad.zip", output.getvalue())})
    assert result.status_code == 422
    record = Dataset.objects.get()
    assert record.status == "failed" and "配对" in record.error
    assert not paths.dataset_yaml_path(record.id).exists()


def test_preview_urls_preserve_hash_percent_and_chinese_filenames(web, inline_import):
    client, _, _ = web
    original = client.get("/api/datasets/template").content; output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(output, "w") as target:
        for name in source.namelist(): target.writestr(name.replace("sample_", "路面#100%_"), source.read(name))
    result = client.post("/api/datasets", data={"name": "中文样本", "mode": "archive"}, files={"files": ("dataset.zip", output.getvalue())})
    assert result.status_code == 202
    rows = client.get(f"/api/datasets/{result.json()['id']}/previews").json()
    assert rows and all("%23" in row["image"] for row in rows)
    for row in rows:
        assert client.get(row["image"]).status_code == 200
        assert client.get(row["mask"]).status_code == 200
