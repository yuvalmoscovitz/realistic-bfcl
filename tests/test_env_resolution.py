from __future__ import annotations

from pathlib import Path

import pytest

from realistic_bfcl import pipeline
from realistic_bfcl.common import (
    anthropic_api_key,
    openai_api_key,
    openrouter_api_key,
    set_explicit_env_file,
)


@pytest.fixture(autouse=True)
def reset_explicit_env_file() -> None:
    set_explicit_env_file(None)
    yield
    set_explicit_env_file(None)


def write_env(path: Path, **values: str) -> Path:
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()), encoding="utf-8")
    return path


def test_key_precedence_is_explicit_then_configured_then_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = write_env(tmp_path / "explicit.env", OPENAI_API_KEY="explicit")
    configured = write_env(tmp_path / "configured.env", OPENAI_API_KEY="configured")
    monkeypatch.setenv("REALISTIC_BFCL_ENV_FILE", str(configured))
    monkeypatch.setenv("OPENAI_API_KEY", "process")

    set_explicit_env_file(explicit)

    assert openai_api_key() == "explicit"


def test_configured_file_precedes_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = write_env(tmp_path / "configured.env", OPENAI_API_KEY="configured")
    monkeypatch.setenv("REALISTIC_BFCL_ENV_FILE", str(configured))
    monkeypatch.setenv("OPENAI_API_KEY", "process")

    assert openai_api_key() == "configured"


def test_provider_aliases_are_supported(tmp_path: Path) -> None:
    explicit = write_env(
        tmp_path / "providers.env",
        CLAUDE_API_KEY="anthropic",
        OPEN_ROUTER_API_KEY="openrouter",
    )
    set_explicit_env_file(explicit)

    assert anthropic_api_key() == "anthropic"
    assert openrouter_api_key() == "openrouter"


def test_nonexistent_explicit_file_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.env"

    with pytest.raises(SystemExit, match="Environment file does not exist"):
        set_explicit_env_file(missing)


def test_invalid_env_file_fails_clearly(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.env"
    invalid.write_text("OPENAI_API_KEY\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="line 1 must be KEY=VALUE"):
        set_explicit_env_file(invalid)


def test_env_parser_supports_safe_dotenv_syntax(tmp_path: Path) -> None:
    env_file = tmp_path / "syntax.env"
    env_file.write_text(
        "export OPENAI_API_KEY='quoted=#=value' # comment\n"
        'ANTHROPIC_API_KEY="double#value"\n'
        "OPENROUTER_API_KEY=plain=value # trailing comment\n",
        encoding="utf-8",
    )

    set_explicit_env_file(env_file)

    assert openai_api_key() == "quoted=#=value"
    assert anthropic_api_key() == "double#value"
    assert openrouter_api_key() == "plain=value"


@pytest.mark.parametrize(
    "contents",
    [
        "OPENAI_API_KEY='unmatched\n",
        "OPENAI_API_KEY=first\nOPENAI_API_KEY=second\n",
    ],
)
def test_env_parser_rejects_ambiguous_values(tmp_path: Path, contents: str) -> None:
    env_file = tmp_path / "invalid.env"
    env_file.write_text(contents, encoding="utf-8")

    with pytest.raises(SystemExit, match="unmatched quote|duplicates"):
        set_explicit_env_file(env_file)


def test_missing_required_key_names_all_supported_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REALISTIC_BFCL_ENV_FILE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="Missing OPENAI_API_KEY"):
        openai_api_key()


def test_cli_env_file_is_only_accepted_for_run_bfcl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = write_env(tmp_path / "run.env", OPENAI_API_KEY="explicit")
    monkeypatch.delenv("REALISTIC_BFCL_ENV_FILE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    captured: dict[str, object] = {}

    def capture_key(models: list[str] | None) -> None:
        captured.update(models=models, key=openai_api_key())

    monkeypatch.setattr(pipeline, "run_bfcl", capture_key)

    pipeline.main(["run-bfcl", "--models", "nano", "--env-file", str(env_file)])

    assert captured == {"models": ["nano"], "key": "explicit"}
    with pytest.raises(SystemExit, match="Missing OPENAI_API_KEY"):
        openai_api_key()

    with pytest.raises(SystemExit):
        pipeline.main(["analyze", "--env-file", str(env_file)])


def test_cli_requires_explicit_model_selection(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("REALISTIC_BFCL_MODELS", raising=False)

    with pytest.raises(SystemExit):
        pipeline.main(["run-bfcl"])
    assert "requires --models" in capsys.readouterr().err

    pipeline.main(["run-bfcl", "--dry-run"])


def test_explicit_env_state_is_cleared_after_failed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = write_env(tmp_path / "run.env", OPENAI_API_KEY="explicit")
    monkeypatch.delenv("REALISTIC_BFCL_ENV_FILE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fail(_models: list[str] | None) -> None:
        assert openai_api_key() == "explicit"
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(pipeline, "run_bfcl", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        pipeline.run_stage(
            pipeline.stage_by_name("run-bfcl"),
            dry_run=False,
            env_file=env_file,
        )

    with pytest.raises(SystemExit, match="Missing OPENAI_API_KEY"):
        openai_api_key()


def test_invalid_replacement_clears_previous_explicit_file(tmp_path: Path) -> None:
    first = write_env(tmp_path / "first.env", OPENAI_API_KEY="first")
    set_explicit_env_file(first)
    assert openai_api_key() == "first"

    with pytest.raises(SystemExit, match="does not exist"):
        set_explicit_env_file(tmp_path / "missing.env")

    with pytest.raises(SystemExit, match="Missing OPENAI_API_KEY"):
        openai_api_key()
