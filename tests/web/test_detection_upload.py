"""多类别 VOC/YOLO 上传,检测模型路由与大包边界。"""
import io
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest
import yaml

from od_platform.common import paths
from od_platform.data_pipeline import detection_import
from od_web import datasets
from od_web.dataset_worker import run


def voc_archive(*, bad_box=False, missing=False):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for index in range(9):
            image = io.BytesIO(); Image.new("RGB", (48, 32), (index*20, 40, 90)).save(image, format="PNG")
            stem = f"harbor{index:02d}_0001"
            archive.writestr(f"WSODD USV_dataset/image/{stem}.png", image.getvalue())
            if missing and index == 0: continue
            objects = "".join(f"<object><name>{name}</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>{80 if bad_box else 25}</xmax><ymax>26</ymax></bndbox></object>" for name in ["boat", "ship", "buoy"])
            archive.writestr(f"WSODD USV_dataset/annotation/{stem}.xml", f"<annotation><size><width>48</width><height>32</height></size>{objects}</annotation>")
    return buffer.getvalue()


@pytest.fixture
def detection_web(web, monkeypatch):
    monkeypatch.setattr(datasets, "launch_import", lambda record, source, mode: run(record.id, source, mode))
    return web


def send(client, blob):
    return client.post("/api/datasets", data={"name": "港区水域通航目标检测数据集", "mode": "archive"}, files={"files": ("WSODD.zip", blob)})


def test_voc_multiclass_upload_preview_and_training_recipe(detection_web, monkeypatch):
    client, store, _ = detection_web
    response = send(client, voc_archive())
    assert response.status_code == 202, response.text
    data = response.json(); identifier = data["id"]
    assert data["task"] == "detect" and data["trainable"]
    assert list(data["names"].values()) == ["boat", "buoy", "ship"]
    assert all(set(counts) == {"boat", "buoy", "ship"} for counts in data["quality"]["class_counts"].values())
    configuration = client.get("/api/config", params={"dataset": identifier}).json()["defaults"]
    assert configuration["backend"] == "yolo" and configuration["data"] == identifier
    previews = client.get(f"/api/datasets/{identifier}/previews").json()
    assert previews[0]["annotation_type"] == "boxes"
    original, boxes = [client.get(previews[0][kind]) for kind in ("image", "mask")]
    assert original.status_code == boxes.status_code == 200 and original.content != boxes.content
    submitted = []
    monkeypatch.setattr("od_web.app._hardware", lambda: {"cuda_available": True})
    monkeypatch.setattr(store, "create", lambda doc: submitted.append(doc) or {"id": "a"*32})
    result = client.post("/api/jobs", json={"kind": "training", "config": {"data": identifier}})
    assert result.status_code == 202, result.text
    assert submitted[0]["config"]["backend"] == "yolo" and submitted[0]["config"]["device"] == "cuda"
    assert submitted[0]["dataset_fingerprint"] == data["fingerprint"]


@pytest.mark.parametrize("bad_box,missing", [(True, False), (False, True)])
def test_voc_bad_annotations_never_become_trainable(detection_web, bad_box, missing):
    assert send(detection_web[0], voc_archive(bad_box=bad_box, missing=missing)).status_code == 422


@pytest.mark.parametrize("replacement", ["9 .5 .5 .2 .2", "0 .5 .5 2 2", "0 .5 .5 .2 .2 .1", "0 nan .5 .2 .2"])
def test_invalid_yolo_box_and_segmentation_polygon_are_rejected(detection_web, replacement):
    client = detection_web[0]; original = client.get("/api/datasets/template").content
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(output, "w") as target:
        for name in source.namelist(): target.writestr(name, replacement if name.endswith("harbor00_0001.txt") else source.read(name))
    assert send(client, output.getvalue()).status_code == 422


def test_large_upload_and_extraction_limits_are_configured_and_enforced(detection_web, monkeypatch):
    assert datasets.MAX_UPLOAD == 4 * 1024**3
    assert detection_import.MAX_EXPANDED == 12 * 1024**3
    monkeypatch.setattr(datasets, "MAX_UPLOAD", 10)
    assert "4 GB" in send(detection_web[0], voc_archive()).json()["detail"]


def test_class_map_and_provided_split_duplicates_are_checked(detection_web):
    client, _, _ = detection_web; original = client.get("/api/datasets/template").content
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(output, "w") as target:
        for split in ("train", "val", "test"):
            for suffix, kind in (("png", "images"), ("txt", "labels")):
                target.writestr(f"d/{kind}/{split}/harbor00_0001.{suffix}", source.read(f"demo/{kind}/harbor00_0001.{suffix}"))
        target.writestr("d/data.yaml", "names: [boat, ship, buoy]\n")
    result = send(client, output.getvalue())
    assert result.status_code == 202 and result.json()["status"] == "blocked"
    checks = result.json()["quality"]["results"]
    assert next(row for row in checks if row["name"] == "detection_duplicates")["severity"] == "ERROR"


def test_names_and_labels_remain_multiclass_without_executing_uploaded_config(detection_web):
    client, _, _ = detection_web; original = client.get("/api/datasets/template").content
    output = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(output, "w") as target:
        for name in source.namelist():
            target.writestr(name, "names: [boat, ship, buoy]\npath: /etc\ndownload: touch /tmp/should-not-run\n" if name.endswith("data.yaml") else source.read(name))
    result = send(client, output.getvalue()).json()
    assert result["trainable"]
    document = yaml.safe_load(paths.dataset_yaml_path(result["id"]).read_text(encoding="utf-8"))
    assert "download" not in document and document["path"].startswith("data/processed/ds_")
    manifest = json.loads((paths.PROCESSED_DATA_DIR / result["id"] / "manifest.json").read_text(encoding="utf-8"))
    assert all(set(row["classes"]) == {0, 1, 2} for row in manifest["samples"])
    assert not (Path(document["path"]) / "masks").exists()


@pytest.mark.parametrize("member,allowed", [("data/processed/ds_1111111111111111/labels/train/a.txt", True),
                                            ("data/processed/ds_2222222222222222/labels/train/a.txt", False),
                                            ("data/processed/ds_1111111111111111/../../outside.txt", False)])
def test_remote_detection_bundle_is_restricted_to_requested_dataset(web, member, allowed):
    import tarfile
    from od_web import remote_agent
    _, store, root = web
    directory = store.root / ("a" * 32); directory.mkdir()
    (directory / "request.json").write_text(json.dumps({"config": {"backend": "yolo", "data": "ds_1111111111111111"}}))
    with tarfile.open(directory / "dataset.tar", "w") as archive:
        info = tarfile.TarInfo(member); info.size = 3; archive.addfile(info, io.BytesIO(b"abc"))
    if allowed:
        assert remote_agent.unpack(directory.name)["unpacked"]
        assert (root / member).read_bytes() == b"abc"
    else:
        with pytest.raises(ValueError): remote_agent.unpack(directory.name)
