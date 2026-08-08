from __future__ import annotations

import pytest
from scripts.check_staged_env_files import is_forbidden_env_path


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        "config/.env.production",
        "nested/path/.env.development",
    ],
)
def test_private_env_paths_are_forbidden(path: str) -> None:
    assert is_forbidden_env_path(path)


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        ".env.template",
        "docs/environment.md",
        "config/env.yaml",
    ],
)
def test_documentation_and_templates_are_allowed(path: str) -> None:
    assert not is_forbidden_env_path(path)
