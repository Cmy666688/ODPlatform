#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""training 子系统测试。核心: 四问写成断言, 且第4问(数据身份)确认闭合。"""
from __future__ import annotations

import ast
import inspect
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from od_platform.common import naming
from od_platform.common.naming import run_stem
from od_platform.data_validation.identity import DataIdentity
from od_platform.training import archive as archive_module, result as result_module
from od_platform.training.result import TrainMetrics
from od_platform.training.service import train_yolo


class _FakeMaps:
    """鸭子类型版 numpy maps: 有 size、能按下标取。Py3.12 起 SimpleNamespace 类不可变,
    不能事后塞 __getitem__, 用真类代替(语义相同, 更干净)。"""
    size = 2
    def __getitem__(self, i): return [0.74, 0.10][i]


class FakeYoloResults:
    task = "detect"; fitness = 0.66
    speed = {"preprocess": 1.2, "inference": 8.4, "loss": 0.0, "postprocess": 0.9}
    results_dict = {"metrics/mAP50(B)": 0.87, "metrics/mAP50-95(B)": 0.61}
    names = {0: "tumor", 1: "lesion"}
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.maps = _FakeMaps()


class FakeRun:
    def __init__(self, tmp_path: Path):
        self.run_id = "20260718-121232"
        self.started_at = datetime(2026, 7, 18, 12, 12, 32)
        self.created_at = self.started_at.isoformat(timespec="seconds")
        self.run_dir = tmp_path / "runs" / "training" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
    def artifact_path(self, *parts: str) -> Path:
        p = self.run_dir.joinpath(*parts); p.parent.mkdir(parents=True, exist_ok=True); return p


@pytest.fixture
def fake_run(tmp_path): return FakeRun(tmp_path)


class FakeConfig:
    """最小真配置类。不用 MagicMock: train_yolo 会把 config 交给 log_config_report,
    后者走 type(config).model_fields —— MagicMock 的类没有这张表, 真类才有。"""
    model_fields = {"model": None, "data": None, "experiment_name": None, "epochs": None}
    model = "yolov8n.pt"; data = "MRI_PASCAL"; experiment_name = "smoke"; epochs = 1
    def to_ultralytics_kwargs(self): return {"epochs": 1}
    def to_audit_snapshot(self):
        return {"config_class": "FakeConfig", "values": {"epochs": 1}}


@pytest.fixture
def fake_config(): return FakeConfig()


@pytest.fixture
def fake_merger():
    m = MagicMock()
    m.to_audit_log.return_value = {"overridden_fields": ["epochs"]}
    m.get_metadata.return_value = None     # 溯源表: 全部字段标 DEFAULT, 不编造链路
    return m


# 一个"闭合"的身份对账结果(三方指纹一致 + D4 PASS)
OK_IDENTITY = DataIdentity(ok=True, data_fingerprint="8f2a41c9deadbeef",
                           contract_fingerprint="46405cf7",
                           transform_stamp={"data_fingerprint": "8f2a41c9deadbeef"},
                           validate_stamp={"data_fingerprint": "8f2a41c9deadbeef",
                                           "overall_severity": "PASS", "tool_version": "odp-validate/0.4"})


# ★ 收束: 四问写成断言
def test_audit_answers_four_questions(fake_run, fake_config, fake_merger):
    with patch("od_platform.training.service._run_training",
               return_value=FakeYoloResults(fake_run.run_dir)), \
         patch("od_platform.training.service._resolve_model_arg", return_value="yolov8n.pt"), \
         patch("od_platform.training.service.reconcile_data_identity", return_value=OK_IDENTITY), \
         patch("od_platform.training.service.stamp.write_stamp"):     # 别真写盘
        result = train_yolo(config=fake_config, data_yaml=Path("MRI_PASCAL.yaml"), run=fake_run,
                            stem="train-20260718-121232-MRI_PASCAL-yolov8n",
                            merger=fake_merger, archive=False)
    assert result.success
    audit = json.loads((fake_run.run_dir / "odp_audit.json").read_text(encoding="utf-8"))
    # ① 产物一套
    assert audit["run_id"] == "20260718-121232"
    assert audit["run_id"] in audit["artifacts"]["stem"]
    # ② 参数 + 谁给的
    assert audit["config"]["config_class"] == "FakeConfig"
    assert "overridden_fields" in audit["config_sources"]
    # ③ 结果, 且属于这次
    assert audit["metrics"]["run_id"] == audit["run_id"]
    assert audit["metrics"]["class_map_50_95"]["tumor"] == 0.74
    # ④ ★ 数据身份: 闭合 + 已验证 PASS + 确认了两枚章(而非转抄)
    di = audit["data_identity"]
    assert di["reconciled"] is True and di["validated_pass"] is True
    assert di["transform_stamp_seen"] is True and di["validate_stamp_seen"] is True
    assert di["data_fingerprint"] == "8f2a41c9deadbeef"     # 给 D7 用的锚


def test_unvalidated_data_blocks_training(fake_run, fake_config, fake_merger):
    """D4 章缺失(require_validated 默认 True)-> 拒绝开训, 不浪费 GPU。"""
    blocked = DataIdentity(ok=False, reason="未找到 D4 的 validate 章")
    with patch("od_platform.training.service.reconcile_data_identity", return_value=blocked), \
         patch("od_platform.training.service._run_training") as run_training:
        result = train_yolo(config=fake_config, data_yaml=Path("d.yaml"), run=fake_run,
                            stem="s", merger=fake_merger, archive=False)
    run_training.assert_not_called()
    assert result.success is False and "validate 章" in result.message


def test_created_at_is_run_started_at(fake_run, fake_config, fake_merger):
    with patch("od_platform.training.service._run_training", return_value=FakeYoloResults(fake_run.run_dir)), \
         patch("od_platform.training.service._resolve_model_arg", return_value="yolov8n.pt"), \
         patch("od_platform.training.service.reconcile_data_identity", return_value=OK_IDENTITY), \
         patch("od_platform.training.service.stamp.write_stamp"):
        train_yolo(config=fake_config, data_yaml=Path("d.yaml"), run=fake_run, stem="s",
                   merger=fake_merger, archive=False)
    audit = json.loads((fake_run.run_dir / "odp_audit.json").read_text(encoding="utf-8"))
    assert audit["created_at"] == "2026-07-18T12:12:32"


# 防漂: 断言"模块没有能力做坏事"
def _imports_of(module) -> set:
    """模块【直接 import】了什么。查 AST 而非字符串匹配 —— docstring 里讨论
    'datetime' / 'run_stem' 是在立规矩, 不等于拥有能力; import 了才算有。"""
    tree = ast.parse(inspect.getsource(module))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_naming_module_has_no_clock():
    assert not any("datetime" in m for m in _imports_of(naming))


def test_archive_has_no_clock_no_naming_rule():
    imports = _imports_of(archive_module)
    assert not any("datetime" in m for m in imports)
    assert not any("naming" in m for m in imports)


def test_result_module_has_no_clock():
    assert not any("datetime" in m for m in _imports_of(result_module))


def test_train_metrics_requires_run_id():
    p = inspect.signature(TrainMetrics.from_yolo_results).parameters["run_id"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty


def test_names_share_one_stem(fake_run):
    stem = run_stem(stage="train", run_id=fake_run.run_id, dataset="MRI_PASCAL", model="yolov8n.pt")
    assert stem == "train-20260718-121232-MRI_PASCAL-yolov8n"
    assert f"{stem.split('-')[1]}-{stem.split('-')[2]}" == fake_run.run_id


def test_metrics_use_config_task_when_library_result_omits_it(tmp_path):
    """部分 ultralytics 版本的 metrics 无 task,不能把已知检测任务记成 unknown。"""
    results = SimpleNamespace(save_dir=tmp_path, results_dict={"metrics/mAP50(B)": 0.8})
    metrics = TrainMetrics.from_yolo_results(results, run_id="test", task="detect")
    assert metrics.task == "detect"
    assert metrics.overall["metrics/mAP50(B)"] == 0.8


def test_metrics_prefer_result_task_when_present(tmp_path):
    results = FakeYoloResults(tmp_path)
    metrics = TrainMetrics.from_yolo_results(results, run_id="test", task="segment")
    assert metrics.task == "detect"
