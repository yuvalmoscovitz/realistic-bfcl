from __future__ import annotations

import subprocess
from pathlib import PurePosixPath

ALLOWED_TEMPLATES = {".env.example", ".env.template"}


def is_forbidden_env_path(path: str) -> bool:
    name = PurePosixPath(path).name
    return (name == ".env" or name.startswith(".env.")) and name not in ALLOWED_TEMPLATES


def staged_paths() -> list[str]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--cached",
            "--name-only",
            "--diff-filter=ACMR",
            "-z",
        ],
        check=True,
        capture_output=True,
    )
    return [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]


def main() -> int:
    forbidden = sorted(path for path in staged_paths() if is_forbidden_env_path(path))
    if not forbidden:
        return 0

    print("Refusing to commit private environment files:")
    for path in forbidden:
        print(f"  - {path}")
    print("Use process variables or --env-file with an untracked file instead.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
