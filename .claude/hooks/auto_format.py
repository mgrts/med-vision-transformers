#!/usr/bin/env python3
"""PostToolUse hook (Edit|Write|MultiEdit): run isort + black on an edited .py file.

Keeps the working tree consistent with `make format` (isort --profile black + black,
line-length 99, per pyproject.toml [tool.black]) so diffs stay clean. There is no
.venv in this repo, so the tools are resolved from PATH (where `make format` finds
them). Always exits 0 — formatting is best-effort and never blocks the edit.
"""

import json
import os
import subprocess
import sys


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    file_path = (data.get("tool_input") or {}).get("file_path", "")
    if not file_path or not file_path.endswith(".py"):
        return 0

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    abs_file = os.path.abspath(file_path)
    abs_proj = os.path.abspath(project_dir)
    try:
        inside = os.path.commonpath([abs_file, abs_proj]) == abs_proj
    except ValueError:
        inside = False
    if not inside or not os.path.isfile(abs_file):
        return 0

    venv_bin = os.path.join(project_dir, ".venv", "bin")

    def tool(name):
        p = os.path.join(venv_bin, name)
        return p if os.path.exists(p) else name

    applied = []
    for name, args in (
        ("isort", ["--profile", "black", "--line-length", "99"]),
        ("black", ["--line-length", "99"]),
    ):
        try:
            subprocess.run(
                [tool(name), *args, abs_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            applied.append(name)
        except FileNotFoundError:
            pass

    if applied:
        print(
            f"[auto-format] {' + '.join(applied)} applied to {os.path.relpath(abs_file, abs_proj)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
