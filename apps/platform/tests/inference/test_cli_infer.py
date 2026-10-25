"""odp-infer CLI 单元测试: 参数解析 + 退出码."""
from __future__ import annotations

from pathlib import Path

import pytest

import od_platform.cli.infer as cli
from od_platform.inference import InferResult


class TestParser:
    def test_defaults_are_none(self):
        ns = cli._build_parser().parse_args([])
        assert ns.source is None and ns.model is None and ns.yaml is None
        assert ns.show is None and ns.save is None
        assert ns.conf is None and ns.batch is None

    def test_no_show_sets_false(self):
        ns = cli._build_parser().parse_args(["--no-show"])
        assert ns.show is False

    def test_no_save_sets_false(self):
        ns = cli._build_parser().parse_args(["--no-save"])
        assert ns.save is False

    def test_show_and_no_show_are_exclusive(self):
        with pytest.raises(SystemExit):
            cli._build_parser().parse_args(["--show", "--no-show"])

    def test_classes_accepts_multiple(self):
        ns = cli._build_parser().parse_args(["--classes", "0", "2", "5"])
        assert ns.classes == [0, 2, 5]


class TestNsToCliArgs:
    def test_keeps_explicit_false(self):
        ns = cli._build_parser().parse_args(["--no-show", "--conf", "0.3"])
        args = cli._ns_to_cli_args(ns)
        assert args["show"] is False          # 显式假值必须保留 (D5 纪律)
        assert args["conf"] == 0.3

    def test_drops_none_fields(self):
        ns = cli._build_parser().parse_args([])
        assert cli._ns_to_cli_args(ns) == {}

    def test_cli_only_flags_not_forwarded(self):
        ns = cli._build_parser().parse_args(
            ["--no-viz", "--no-hud", "--warmup", "5", "--log-level", "DEBUG"])
        args = cli._ns_to_cli_args(ns)
        for key in ("no_viz", "no_hud", "warmup", "log_level"):
            assert key not in args


class TestMain:
    def test_success_returns_zero(self, monkeypatch):
        calls = {}

        def fake_infer(**kwargs):
            calls.update(kwargs)
            return InferResult(success=True, output_dir=Path("out"))

        monkeypatch.setattr(cli, "get_logger", lambda **kw: None)
        monkeypatch.setattr(cli, "infer_yolo", fake_infer)

        assert cli.main(["--source", "imgs", "--conf", "0.4", "--no-viz", "--no-hud"]) == 0
        assert calls["beautify"] is False
        assert calls["show_info"] is False
        assert calls["cli_args"]["conf"] == 0.4
        # CLI 不传接缝参数 → service 走默认
        assert "output_sink" not in calls
        assert "hooks" not in calls
        assert "cancel_token" not in calls

    def test_failure_returns_one(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "get_logger", lambda **kw: None)
        monkeypatch.setattr(
            cli, "infer_yolo",
            lambda **kw: InferResult(success=False, output_dir=Path("out"),
                                     error="boom", log_path=Path("x.log")),
        )
        assert cli.main([]) == 1
        err = capsys.readouterr().err
        assert "推理失败" in err and "boom" in err
