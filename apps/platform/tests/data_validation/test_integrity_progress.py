"""旧 CLI 的完整解码及有界进度回调。"""
from pathlib import Path
from types import SimpleNamespace
import io
import tarfile
import zipfile

import pytest
from PIL import Image

from od_platform.data_pipeline.import_dataset import extract_archive
from od_platform.data_validation.checks.image_integrity import validate_image_integrity


def test_integrity_detects_broken_pixels_and_reports_progress(tmp_path):
    good = tmp_path / "good.png"; Image.new("RGB", (16,16)).save(good)
    bad = tmp_path / "bad.png"; bad.write_bytes(good.read_bytes()[:40])
    events = []; snapshot = SimpleNamespace(splits=["train"], images_per_split={"train":(good,bad)})
    result = validate_image_integrity(SimpleNamespace(snapshot=snapshot, on_progress=lambda *event: events.append(event)))
    assert result.severity == "ERROR" and result.details["count"] == 1
    assert events[0][1] == 0 and events[-1][1:] == (2,2)


@pytest.mark.parametrize("kind", ["zip", "tar"])
def test_archive_progress_includes_exact_terminal_count(tmp_path, kind):
    file = tmp_path / ("samples." + kind)
    if kind == "zip":
        with zipfile.ZipFile(file, "w") as archive:
            for index in range(3): archive.writestr(f"{index}.txt", "data")
    else:
        with tarfile.open(file, "w") as archive:
            for index in range(3):
                member = tarfile.TarInfo(f"{index}.txt"); member.size = 4; archive.addfile(member, io.BytesIO(b"data"))
    events = []; destination = tmp_path / "out"
    extract_archive(Path(file), destination, on_progress=lambda *event: events.append(event))
    assert events[-1] == (3,3) and len(list(destination.glob("*.txt"))) == 3
