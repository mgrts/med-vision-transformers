#!/usr/bin/env python3
"""PreToolUse hook (Bash): block destructive git operations.

Blocks (exit 2 -> tool call denied, stderr shown to the agent):
  - force pushes (git push --force / -f)   [--force-with-lease is allowed]
  - git push --no-verify
  - git commit --no-verify / -n
  - git commit carrying Claude/AI author attribution (Co-Authored-By / --author /
    "Generated with Claude" / @anthropic.com) — project convention: Claude is NEVER
    listed among commit authors
  - git reset --hard
  - deleting the main branch (local `git branch -D main` or remote `push --delete`/`:main`)

The hook only gates the agent; the user can still run any of these manually.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gitutil import has_short_flag, is_main_ref, parse_git_invocations  # noqa: E402

# AI/Claude AUTHORSHIP attribution only — deliberately NOT a bare "claude" match, so a
# legitimate topical subject like `feat(claude): ...` is never blocked.
AI_ATTRIBUTION = re.compile(
    r"co-authored-by:[^\r\n]*(?:claude|anthropic)"
    r"|--author[=\s][^\r\n]*(?:claude|anthropic)"
    r"|noreply@anthropic\.com"
    r"|@anthropic\.com"
    r"|generated with[^\r\n]*claude",
    re.IGNORECASE,
)


def deny(what: str, why: str) -> int:
    sys.stderr.write(
        f"⛔ guard-git blocked this command: {what}\n"
        f"Reason: {why}\n"
        "If this is genuinely intended, run it yourself in a terminal "
        "(the hook only gates the agent).\n"
    )
    return 2


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0
    cmd = (data.get("tool_input") or {}).get("command", "")
    if not cmd:
        return 0

    # Block Claude/AI author attribution on ANY git commit, independent of how the
    # message is supplied (multiple -m, heredoc, embedded newlines). Uses a loose
    # commit detector so a newline-containing message can't dodge the strict parser.
    if AI_ATTRIBUTION.search(cmd) and re.search(r"\bgit\b[^\n]*\bcommit\b", cmd):
        return deny(
            "git commit with Claude/AI author attribution",
            "This project's convention is that Claude is NEVER listed among commit "
            "authors. Remove any 'Co-Authored-By: Claude/Anthropic' trailer, '--author' "
            "set to Claude/Anthropic, '@anthropic.com' address, or 'Generated with "
            "Claude' line from the commit message.",
        )

    for sub, subargs in parse_git_invocations(cmd):
        if sub == "push":
            # --force (token-exact, so --force-with-lease is untouched) or bare -f
            if "--force" in subargs or has_short_flag(subargs, "f"):
                return deny(
                    "git push --force / -f",
                    "Force-push can overwrite remote history. Use "
                    "'git push --force-with-lease' manually if you truly need it.",
                )
            if "--no-verify" in subargs:
                return deny(
                    "git push --no-verify", "Skipping hooks defeats the commit/push gates."
                )
            deletes_remote_main = (
                "--delete" in subargs and any(is_main_ref(a) for a in subargs)
            ) or any(re.fullmatch(r":(?:refs/heads/)?main", a) for a in subargs)
            if deletes_remote_main:
                return deny(
                    "deleting remote 'main'", "Refusing to delete the main branch on origin."
                )
        elif sub == "reset":
            if "--hard" in subargs:
                return deny(
                    "git reset --hard",
                    "Discards uncommitted work irreversibly. Stash or commit first, or run it yourself.",
                )
        elif sub == "commit":
            if "--no-verify" in subargs or has_short_flag(subargs, "n"):
                return deny(
                    "git commit --no-verify / -n",
                    "Bypasses pre-commit (isort/flake8 + large-file/secret scan).",
                )
        elif sub == "branch":
            deletes = (
                "-d" in subargs
                or "-D" in subargs
                or has_short_flag(subargs, "d")
                or has_short_flag(subargs, "D")
            )
            if deletes and any(is_main_ref(a) for a in subargs):
                return deny("git branch -D main", "Refusing to delete the local main branch.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
