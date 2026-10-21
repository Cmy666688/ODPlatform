#!/usr/bin/env python
# @FileName  : test_reset_project.py
# @Author    : ODPlatform Team
# @Project   : ODPlatform
# @Function  : reset_project(D2) 的独立单元测试
#
# 覆盖四层防御:
#   1. 白名单契约——get_dirs_to_reset 不含 git 追踪目录 / raw 数据 / META 目录
#   2. 黑名单兜底——is_protected 对保护目录及其祖先返回 True
#   3. 行为门——确认词精确匹配 / dry-run 默认不删 / --dry-run 压过 --yes
#   4. 删除与扫描——只删目标目录,受保护目录拒删
#
# 安全约定:所有涉及删除的用例都把 ROOT_DIR monkeypatch 到 pytest 的
# tmp_path,绝不对真实项目目录做任何删除。

from pathlib import Path

import pytest

from od_platform.cli import reset_project as rp
from od_platform.common import paths


# ============================================================
# 1. 白名单契约
# ============================================================
class TestResetWhitelist:
    """get_dirs_to_reset() 的边界——reset 只认这份白名单"""

    def test_whitelist_only_contains_expected_dirs(self):
        expected = {
            paths.PROCESSED_DATA_DIR,
            paths.RUNS_DIR,
            paths.TRAINED_MODELS_DIR,
            paths.LOGGING_DIR,
        }
        assert set(paths.get_dirs_to_reset()) == expected

    def test_whitelist_excludes_git_tracked_dirs(self):
        """git 追踪目录(代码/文档/测试/配置/脚本)绝不能进 reset 清单"""
        reset_list = paths.get_dirs_to_reset()
        for git_dir in [
            paths.ROOT_DIR / "apps",
            paths.APP_DIR / "src",
            paths.SCRIPTS_DIR,
            paths.DOCS_DIR,
            paths.UNIT_TEST_DIR,
            paths.CONFIGS_DIR,
            paths.ROOT_DIR / ".git",
        ]:
            assert git_dir not in reset_list, f"{git_dir} 不应出现在 reset 清单中"

    def test_whitelist_excludes_raw_and_pretrained(self):
        """原始数据和预训练权重绝不能进 reset 清单"""
        reset_list = paths.get_dirs_to_reset()
        assert paths.RAW_DATA_DIR not in reset_list
        assert paths.PRETRAINED_MODELS_DIR not in reset_list

    def test_meta_dir_not_in_reset_list(self):
        """审计命名空间绝不能进 reset 清单(自指 bug 防线)"""
        reset_list = paths.get_dirs_to_reset()
        assert paths.META_DIR not in reset_list
        assert paths.META_LOGGING_DIR not in reset_list

    def test_meta_logging_dir_in_init_list(self):
        """META_LOGGING_DIR 由 init 负责创建(reset 永远只读它)"""
        assert paths.META_LOGGING_DIR in paths.get_dirs_to_initialize()

    def test_reset_list_is_subset_of_init_list_parent_space(self):
        """reset 清单中每个目录都是 init 清单的成员(能 init 的运行时目录才谈得上 reset)"""
        init_set = set(paths.get_dirs_to_initialize())
        for d in paths.get_dirs_to_reset():
            assert d in init_set, f"{d} 在 reset 清单但不在 init 清单"


# ============================================================
# 2. 黑名单兜底
# ============================================================
class TestIsProtected:
    """is_protected() 的双向检查——本身 + 祖先"""

    def test_all_protected_dirs_are_protected(self):
        for d in paths.PROTECTED_DIRS:
            assert paths.is_protected(d), f"{d} 应受保护"

    def test_ancestor_of_protected_is_protected(self):
        """删祖先 = 连带删受保护目录,所以祖先也要拦(堵死删 ROOT_DIR 的路)"""
        assert paths.is_protected(paths.ROOT_DIR)
        assert paths.is_protected(paths.ROOT_DIR.parent.parent)  # 比 ROOT 更高层的祖先

    def test_reset_targets_not_protected(self):
        """白名单成员不应被黑名单误拦(否则 reset 永远空转)"""
        for d in paths.get_dirs_to_reset():
            assert not paths.is_protected(d), f"{d} 不应被误判为受保护"

    def test_meta_dir_is_protected(self):
        """审计目录必须同时出现在 PROTECTED_DIRS 中(双保险)"""
        assert paths.META_DIR in paths.PROTECTED_DIRS
        assert paths.META_LOGGING_DIR in paths.PROTECTED_DIRS

    def test_arbitrary_tmp_dir_not_protected(self, tmp_path):
        assert not paths.is_protected(tmp_path)


# ============================================================
# 3. 行为门:确认词 / dry-run 优先级
# ============================================================
class TestConfirmGate:
    """确认词必须精确匹配大写 RESET——任何其他输入都取消"""

    def test_exact_reset_accepted(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda *a: "RESET")
        assert rp._confirm(4) is True

    @pytest.mark.parametrize("bad", ["reset", "Reset", "", " y ", "yes", "RESET！"])
    def test_wrong_input_rejected(self, monkeypatch, bad):
        monkeypatch.setattr("builtins.input", lambda *a: bad)
        assert rp._confirm(4) is False

    def test_keyboard_interrupt_rejected(self, monkeypatch):
        def raise_interrupt(*a):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", raise_interrupt)
        assert rp._confirm(4) is False

    def test_eof_rejected(self, monkeypatch):
        def raise_eof(*a):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        assert rp._confirm(4) is False


class TestFormatSize:
    """二进制单位换算(1024 进制)"""

    @pytest.mark.parametrize("size, expected", [
        (0, "0 B"),
        (512, "512 B"),
        (1024, "1.00 KiB"),
        (1024 ** 2, "1.00 MiB"),
        (3 * 1024 ** 3, "3.00 GiB"),
    ])
    def test_format_size(self, size, expected):
        assert rp._format_size(size) == expected


# ============================================================
# 4. 扫描 / 删除 / 主流程
# ============================================================
class TestScanAndDelete:
    """扫描统计与单目录删除(全部在 tmp_path 下进行)"""

    def _make_dir_with_files(self, base: Path, n: int = 3) -> Path:
        d = base / "fake_runs"
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"file_{i}.txt").write_text(f"data {i}")
        return d

    def test_scan_targets_counts_files_and_bytes(self, tmp_path, monkeypatch):
        d = self._make_dir_with_files(tmp_path, n=4)
        monkeypatch.setattr(rp, "get_dirs_to_reset", lambda: [d])
        deletable, skipped = rp._scan_targets()
        assert len(deletable) == 1
        path, count, size = deletable[0]
        assert path == d
        assert count == 4
        assert size == sum((d / f"file_{i}.txt").stat().st_size for i in range(4))
        assert skipped == []

    def test_scan_targets_skips_nonexistent(self, tmp_path, monkeypatch):
        ghost = tmp_path / "not_exist"
        monkeypatch.setattr(rp, "get_dirs_to_reset", lambda: [ghost])
        deletable, skipped = rp._scan_targets()
        assert deletable == []
        assert skipped == [ghost]

    def test_delete_one_removes_dir(self, tmp_path, monkeypatch):
        d = self._make_dir_with_files(tmp_path)
        monkeypatch.setattr(rp, "ROOT_DIR", tmp_path)
        assert rp._delete_one(d, idx=1, total=1, file_count=3, size=24) is None
        assert not d.exists()

    def test_delete_one_refuses_protected_dir(self, tmp_path, monkeypatch):
        """受保护目录即使被塞进删除队列,删除前检查也要拦下"""
        monkeypatch.setattr(rp, "ROOT_DIR", tmp_path)
        reason = rp._delete_one(paths.SCRIPTS_DIR, idx=1, total=1, file_count=0, size=0)
        assert reason == "受保护目录"
        assert paths.SCRIPTS_DIR.exists()  # 原目录安然无恙

    def test_execute_delete_reports_success(self, tmp_path, monkeypatch):
        d1 = self._make_dir_with_files(tmp_path / "a")
        d2 = self._make_dir_with_files(tmp_path / "b")
        monkeypatch.setattr(rp, "ROOT_DIR", tmp_path)
        rp._execute_delete([(d1, 3, 24), (d2, 3, 24)])
        assert not d1.exists()
        assert not d2.exists()


class TestResetProjectFlow:
    """主流程:默认 dry-run 不删 / --dry-run 压过 --yes / --yes --force 真删"""

    def _setup_fake_target(self, tmp_path, monkeypatch):
        d = tmp_path / "fake_runs"
        d.mkdir()
        (d / "artifact.txt").write_text("runtime artifact")
        monkeypatch.setattr(rp, "get_dirs_to_reset", lambda: [d])
        monkeypatch.setattr(rp, "ROOT_DIR", tmp_path)
        # _print_plan 末尾会打印"不会被动"的目录(relative_to ROOT_DIR),
        # ROOT_DIR 被 patch 后这两个常量必须同步 patch,否则 relative_to 抛 ValueError
        monkeypatch.setattr(rp, "RAW_DATA_DIR", tmp_path / "data" / "raw")
        monkeypatch.setattr(rp, "PRETRAINED_MODELS_DIR", tmp_path / "models" / "pretrained")
        return d

    def test_default_is_dry_run_and_keeps_dirs(self, tmp_path, monkeypatch):
        d = self._setup_fake_target(tmp_path, monkeypatch)
        assert rp.reset_project() == 0
        assert d.exists()  # 默认行为只打印不删除

    def test_explicit_dry_run_overrides_yes(self, tmp_path, monkeypatch):
        """--dry-run 与 --yes 同时给出时,以 --dry-run 为准"""
        d = self._setup_fake_target(tmp_path, monkeypatch)
        assert rp.reset_project(yes=True, dry_run=True) == 0
        assert d.exists()

    def test_yes_force_deletes_and_returns_zero(self, tmp_path, monkeypatch):
        d = self._setup_fake_target(tmp_path, monkeypatch)
        assert rp.reset_project(yes=True, force=True) == 0
        assert not d.exists()

    def test_yes_with_cancelled_confirm_keeps_dirs(self, tmp_path, monkeypatch):
        d = self._setup_fake_target(tmp_path, monkeypatch)
        monkeypatch.setattr("builtins.input", lambda *a: "reset")  # 小写,取消
        assert rp.reset_project(yes=True, force=False) == 1
        assert d.exists()

    def test_empty_project_returns_zero(self, tmp_path, monkeypatch):
        ghost = tmp_path / "not_exist"
        monkeypatch.setattr(rp, "get_dirs_to_reset", lambda: [ghost])
        monkeypatch.setattr(rp, "ROOT_DIR", tmp_path)
        assert rp.reset_project(yes=True, force=True) == 0


class TestRestoreGitTracked:
    """删除目标目录后,其中的 git 追踪文件(如 README.md)必须被恢复

    场景:.gitignore 规则 `runs/*` + `!runs/README.md` 刻意让 README 进 git;
    reset 的契约是"回到 git 刚 clone 的状态"——那里面【有】这个 README。
    """

    def test_delete_one_restores_tracked_readme(self, tmp_path, monkeypatch):
        import subprocess as sp

        target = tmp_path / "fake_runs"
        target.mkdir()
        (target / "README.md").write_text("# runs\n")
        (target / "runtime.txt").write_text("runtime artifact")

        sp.run(["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True)
        sp.run(["git", "add", "fake_runs/README.md"],
               cwd=tmp_path, check=True, capture_output=True)

        monkeypatch.setattr(rp, "ROOT_DIR", tmp_path)
        assert rp._delete_one(target, idx=1, total=1, file_count=2, size=32) is None
        assert not (target / "runtime.txt").exists()   # 运行时产物被删
        assert (target / "README.md").exists()         # git 追踪文件被恢复
        assert (target / "README.md").read_text() == "# runs\n"

    def test_delete_one_without_git_degrades_gracefully(self, tmp_path, monkeypatch):
        """非 git 环境下恢复静默降级,删除本身仍然成功"""
        target = tmp_path / "fake_runs"
        target.mkdir()
        (target / "runtime.txt").write_text("x")
        monkeypatch.setattr(rp, "ROOT_DIR", tmp_path)
        assert rp._delete_one(target, idx=1, total=1, file_count=1, size=1) is None
        assert not target.exists()


class TestAuditContext:
    """审计上下文必须包含五要素——出事时的第一份证据"""

    def test_audit_context_fields(self):
        ctx = rp._audit_context()
        assert set(ctx.keys()) == {"user", "pid", "git_rev", "argv", "cwd"}
        assert ctx["user"]
        assert isinstance(ctx["pid"], int)
        assert ctx["git_rev"]  # 本仓库在 git 管理下,应能取到 rev
