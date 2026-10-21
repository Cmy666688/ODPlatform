# tests — 跨端(E2E)测试的家

放跨端端到端测试。

- 单端的单元测试放在各自端内(如 `apps/platform/tests/`)
- 这里只放需要多个端协同的 E2E 测试

参考: 仓库根 `pyproject.toml` 的 `[tool.pytest.ini_options]`。
