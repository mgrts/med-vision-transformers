#!/usr/bin/env python3
"""Stop hook: byte-compile src/ when the turn touched Python source.

This repo has no test suite, so the cheap, always-available correctness gate is
``python -m compileall`` — it catches syntax errors / unparseable files introduced
during the turn (stdlib only, no third-party deps required). Runs only if there are
uncommitted changes to .py files under src/. On failure it blocks the stop once
(exit 2) so the agent sees and fixes the breakage; it does NOT re-block if it was
itself the cause of the previous stop (stop_hook_active), to avoid loops.

This is intentionally NOT a runtime check — use the /verify skill for a smoke
1-epoch/2-fold training run. Disable by removing the "Stop" entry from
.claude/settings.json.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gitutil import status_paths  # noqa: E402


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    # Avoid loops: if the previous stop was already triggered by this hook, let it stop.
    if data.get("stop_hook_active"):
        return 0

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()

    touched = [
        p for p in status_paths(project_dir) if p.endswith(".py") and p.startswith("src/")
    ]
    if not touched:
        return 0

    python = os.path.join(project_dir, ".venv", "bin", "python")
    python = python if os.path.exists(python) else sys.executable
    proc = subprocess.run(
        [python, "-m", "compileall", "-q", os.path.join(project_dir, "src")],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        print("[compile-check] src/ compiles cleanly.")
        return 0

    tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
    sys.stderr.write("⛔ src/ has a syntax/compile error after your changes (auto-run on stop):\n")
    sys.stderr.write("\n".join(tail[-25:]) + "\n")
    sys.stderr.write(
        '\nFix the compile error before finishing, or remove the "Stop" hook in '
        ".claude/settings.json to disable this gate.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
