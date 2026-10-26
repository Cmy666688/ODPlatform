"""WSODD 检测接入:冻结划分、数据身份、模型任务和远程边界。"""
import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest

from od_platform.common import paths
from od_platform.data_pipeline import wsodd


def archive(path, *, bad_box=False, bad_size=False):
    with zipfile.ZipFile(path, "w") as target:
        for index in range(12):
            pixels = np.random.default_rng(index).integers(0, 255, (32, 40, 3), dtype=np.uint8)
            buffer = io.BytesIO(); Image.fromarray(pixels).save(buffer, format="JPEG")
            target.writestr(f"WSODD USV_dataset/image/{index}.jpg", buffer.getvalue())
            objects = "".join(f"<object><name>{name}</name><bndbox><xmin>1</xmin><ymin>2</ymin><xmax>{80 if bad_box else 30}</xmax><ymax>29</ymax></bndbox></object>" for name in wsodd.NAMES)
            document = f"<annotation><folder>sequence{index//3}</folder><filename>{index}.jpg</filename><size><width>{41 if bad_size else 40}</width><height>32</height></size>{objects}</annotation>"
            target.writestr(f"WSODD USV_dataset/annotation/{index}.xml", document)


def prepare(web, **kwargs):
    path = web[2] / "wsodd.zip"; archive(path, **kwargs)
    paths.DATASET_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    return wsodd.prepare(path)


def test_detection_pairs_groups_fingerprint_and_preview(web):
    client, _, root = web
    report = prepare(web)
    assert report["overall_severity"] == "WARNING"
    manifest = json.loads((root / "data/processed/wsodd/manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["samples"]) == 12
    groups = {}
    for row in manifest["samples"]:
        assert groups.setdefault(row["group"], row["split"]) == row["split"]
    overview = client.get("/api/overview").json()
    assert overview["dataset"]["id"] == "wsodd" and overview["dataset"]["task"] == "detect"
    assert overview["dataset"]["trainable"] and sum(overview["dataset"]["splits"].values()) == 12
    previews = client.get("/api/datasets/wsodd/previews").json()
    assert previews and client.get(previews[0]["mask"]).status_code == 200
    assert client.get(previews[0]["image"]).headers["Content-Type"] == "image/jpeg"


@pytest.mark.parametrize("option", ["bad_box", "bad_size"])
def test_voc_errors_are_not_silently_clipped_or_resized(web, option):
    with pytest.raises(ValueError): prepare(web, **{option: True})
    assert not paths.dataset_yaml_path("wsodd").exists()


def test_changed_detection_label_blocks_training_even_with_gpu(web, monkeypatch):
    client, store, root = web; prepare(web)
    label = next((root / "data/processed/wsodd/labels").rglob("*.txt"))
    label.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    monkeypatch.setattr("od_web.app._hardware", lambda: {"cuda_available": True})
    monkeypatch.setattr(store, "create", lambda _: pytest.fail("改动身份后不能开训"))
    response = client.post("/api/jobs", json={"kind": "training", "config": {"backend": "yolo", "data": "wsodd"}})
    assert response.status_code == 422
    assert wsodd.validate(paths.dataset_yaml_path("wsodd"))["overall_severity"] == "ERROR"


def test_wsodd_submission_forces_remote_cuda_and_uses_detection_recipe(web, monkeypatch):
    client, store, _ = web; prepare(web)
    monkeypatch.setattr("od_web.app._hardware", lambda: {"cuda_available": True})
    requests = []
    monkeypatch.setattr(store, "create", lambda document: requests.append(document) or {"id": "1"*32})
    response = client.post("/api/jobs", json={"kind": "training", "config": {"data": "wsodd", "device": "cpu"}})
    assert response.status_code == 202, response.text
    assert requests[0]["config"]["backend"] == "yolo" and requests[0]["config"]["device"] == "cuda"
    assert requests[0]["config"]["epochs"] == 50 and requests[0]["dataset_fingerprint"]
    assert client.post("/api/jobs", json={"kind": "training", "config": {"backend": "mixercseg", "data": "wsodd"}}).status_code == 422
    defaults = client.get("/api/config").json()["defaults"]
    assert defaults["architecture"] == "yolov8n.pt" and defaults["imgsz"] == 640


def test_deterministic_groups_do_not_depend_on_row_order():
    rows = [{"group": f"g{index//4}", "classes": list(range(14))} for index in range(24)]
    assert wsodd.assign_groups(rows) == wsodd.assign_groups(list(reversed(rows)))
    assert wsodd.source_group("192.168.0.64_01_20200908130244888", "annotator1") == wsodd.source_group("192.168.0.64_01_20200908130245999", "annotator2")


def test_image_cache_invalidates_after_jpeg_content_changes(web):
    prepare(web)
    image = next((paths.PROCESSED_DATA_DIR / "wsodd/images").rglob("*.jpg"))
    image.write_bytes(image.read_bytes() + b"changed-source")
    assert wsodd.validate(paths.dataset_yaml_path("wsodd"), use_cache=True)["overall_severity"] == "ERROR"
    assert wsodd.validate(paths.dataset_yaml_path("wsodd"), use_cache=False)["overall_severity"] == "ERROR"
