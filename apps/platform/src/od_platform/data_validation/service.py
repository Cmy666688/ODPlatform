# apps/platform/src/od_platform/data_validation/service.py
"""质检服务层:跑全部 check 并聚合。
唯一的 except Exception 护栏在这里:单 check 崩溃降级为 ERROR 结果,不拖垮整场质检。"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import yaml

from od_platform.common import paths, stamp
from od_platform.common.lineage import compute_data_fingerprint
from od_platform.data_validation import VALIDATOR_VERSION
from od_platform.data_validation.checks._lineage import load_manifest_from_yaml
from od_platform.data_validation.registry import (
    CheckContext, CheckEntry, CheckResult, CheckSeverity, get_all_checks,
)
from od_platform.data_validation.report import ValidationReport
from od_platform.data_validation.snapshot import build_snapshot


def _safe_run_one(entry: CheckEntry, ctx: CheckContext) -> CheckResult:
    try:
        return entry.func(ctx)
    except Exception as e:                          # noqa: BLE001 —— 护栏,必须宽
        return CheckResult(entry.name, CheckSeverity.ERROR,
                           f"check 自身崩溃: {type(e).__name__}: {e}",
                           {"exception": repr(e)})


def run_all_checks(ctx: CheckContext) -> List[CheckResult]:
    return [_safe_run_one(e, ctx) for e in get_all_checks()]


def validate_dataset(yaml_path, task_type: Optional[str] = None,
                     run_id: Optional[str] = None,
                     write_report: bool = True) -> ValidationReport:
    """端到端:快照 → 全量 check → 报告(logger + json/csv)。"""
    t0 = time.perf_counter()
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    yaml_path = Path(yaml_path)
    try:
        document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        document = None                          # 坏 YAML 交回原快照流程报告 ERROR
    if isinstance(document, dict) and document.get("format") == "semantic_masks":
        from od_platform.data_validation.semantic import validate_semantic_dataset
        return validate_semantic_dataset(yaml_path, run_id=run_id, write_report=write_report)
    snapshot = build_snapshot(yaml_path, task_type=task_type)
    results = run_all_checks(CheckContext(yaml_path=yaml_path, snapshot=snapshot))
    report = ValidationReport(run_id=run_id, snapshot=snapshot, results=results,
                              duration_seconds=time.perf_counter() - t0)
    report.render_to_logger()
    if write_report:
        out_dir = paths.validation_run_dir(run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        report.write_json(out_dir / "report.json")
        report.write_results_csv(out_dir / "results.csv")

    # ★ 盖 validate 章: 我(D4)验过这批数据。仅在【能定位到 manifest】时盖 ——
    #   没有 manifest 的数据集(非 odp-transform 产出)谈不上流转卡, 跳过。
    manifest, _err, _odp = load_manifest_from_yaml(snapshot.yaml_data)
    if manifest is not None:
        # data_fingerprint 从【manifest.samples 自己算】—— 不读 D3 的章、不抄它的指纹
        data_fp = compute_data_fingerprint(manifest.samples)
        stamp.write_stamp(
            stage_file=stamp.STAMP_VALIDATE,
            data_fingerprint=data_fp,
            run_id=run_id,
            tool_version=VALIDATOR_VERSION,
            payload={
                "contract_fingerprint": manifest.contract_fingerprint,
                "overall_severity": report.overall_severity,
                "n_checks": len(report.results),
                "report_ref": f"runs/data_validation/{run_id}/report.json",
            },
        )
    return report
