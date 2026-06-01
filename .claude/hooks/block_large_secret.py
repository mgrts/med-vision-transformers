#!/usr/bin/env python3
"""PreToolUse hook (Bash): refuse to add/commit large artifacts or secrets.

Mirrors the repo's pre-commit safety net (check-added-large-files maxkb=10000,
detect-private-key) at the agent layer, and additionally guards the artifact dirs
(data/ models/ mlruns/ reports/ notebooks/) and obvious secret/binary/medical-image
files. Only inspects `git add` and `git commit`; everything else passes through.
Exit 2 blocks.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gitutil import (  # noqa: E402
    diff_paths,
    has_short_flag,
    parse_git_invocations,
    status_paths,
)

SIZE_LIMIT = 10 * 1024 * 1024  # 10 MB, matches check-added-large-files --maxkb=10000
RISKY_DIR = re.compile(r"^(data|models|mlruns|reports|notebooks)(/|$)")
RISKY_FILE = re.compile(
    r"(^|/)(mlruns\.db|id_rsa|credentials\.json)$"
    r"|\.(pem|key|p12|pfx|pkl|pt|pth|ckpt|safetensors|h5|joblib|npy|npz|nii|gz|ipynb|db)$"
    r"|(^|/)\.env(\.|$)"
)

PROJECT_DIR = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()


def expand(paths):
    files = []
    for p in paths:
        ap = p if os.path.isabs(p) else os.path.join(PROJECT_DIR, p)
        if os.path.isdir(ap):
            for root, _, names in os.walk(ap):
                files.extend(os.path.join(root, n) for n in names)
        else:
            files.append(ap)
    return files


def violations(abs_files):
    bad = []
    for f in abs_files:
        if not os.path.isfile(f):
            continue
        base = os.path.basename(f)
        # Legitimate tracked placeholders / env templates (committed on purpose).
        if base == ".gitkeep" or base in (".env.example", ".env.sample", ".env.template"):
            continue
        rel = os.path.relpath(f, PROJECT_DIR)
        if RISKY_DIR.match(rel):
            bad.append(
                (rel, "inside an artifact dir (data/ models/ mlruns/ reports/ notebooks/) — keep it out of git")
            )
            continue
        if RISKY_FILE.search(rel):
            bad.append((rel, "looks like a secret, dataset, model/binary, or medical-image artifact"))
            continue
        try:
            size = os.path.getsize(f)
        except OSError:
            size = 0
        if size > SIZE_LIMIT:
            bad.append((rel, f"{size // (1024 * 1024)} MB exceeds the 10 MB limit"))
    return bad


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    cmd = (data.get("tool_input") or {}).get("command", "")
    if not cmd:
        return 0

    invocations = parse_git_invocations(cmd)
    targets = []
    for sub, subargs in invocations:
        if sub == "add":
            explicit = [t for t in subargs if not t.startswith("-")]
            add_all = (
                any(f in subargs for f in ("-A", "--all", "-u", "--update"))
                or has_short_flag(subargs, "A")
                or "." in explicit
            )
            if explicit and not add_all:
                targets += expand(explicit)
            else:
                targets += expand(status_paths(PROJECT_DIR))
        elif sub == "commit":
            targets += expand(diff_paths(PROJECT_DIR, ["diff", "--cached", "--name-only"]))
            if "--all" in subargs or has_short_flag(subargs, "a"):
                targets += expand(diff_paths(PROJECT_DIR, ["diff", "--name-only"]))

    if not invocations or not targets:
        return 0

    targets = list(dict.fromkeys(os.path.abspath(t) for t in targets))
    bad = violations(targets)
    if not bad:
        return 0

    sys.stderr.write("⛔ block-large-secret refused this git operation. Offending paths:\n")
    for rel, why in bad[:20]:
        sys.stderr.write(f"  - {rel}: {why}\n")
    if len(bad) > 20:
        sys.stderr.write(f"  ... and {len(bad) - 20} more\n")
    sys.stderr.write(
        "\nThese should not enter version control. Options:\n"
        "  - Unstage: git restore --staged <path>\n"
        "  - Keep data/, models/, mlruns/, reports/, notebooks/ in .gitignore (do not 'git add -f')\n"
        "  - For a genuinely needed large file, the user can add it manually.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
