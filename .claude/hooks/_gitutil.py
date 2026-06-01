"""Shared helpers for the git-aware PreToolUse hooks.

Parses a (possibly compound) shell command into its individual `git`
invocations so the guards key off the actual subcommand and flags rather than
substring-matching the whole command line.

Design notes (these matter for both correctness and security):
- The WHOLE command is tokenized with ``shlex`` FIRST, so a quoted commit
  message (e.g. ``-m "fix | reset --hard thing"``) is one opaque token and is
  never re-split into spurious flags/separators.
- ``git`` is detected as the command word even when glued to shell punctuation
  — subshells ``(git ...)``, command substitution ``$(git ...)`` / backticks,
  and ``VAR=$(git ...)`` — so the guards do not fail open on those wrappers.
- ``bash -c "<payload>"`` / ``sh -c '...'`` payloads are re-parsed recursively.
- If ``shlex`` cannot parse the command at all (unbalanced quotes), we return no
  invocations rather than guessing — fail-open on parse failure to avoid
  blocking legitimate commands, since the dangerous wrappers above parse fine.
"""

import re
import shlex
import subprocess

# git global options that consume the following token as their argument
_GLOBAL_WITH_ARG = {
    "-C",
    "-c",
    "--git-dir",
    "--work-tree",
    "--namespace",
    "--exec-path",
    "--super-prefix",
}

# tokens that end one command and start another
_SEPARATORS = {"&&", "||", ";", "|", "&", "(", ")", "{", "}"}
_SHELLS = {"bash", "sh", "zsh", "dash", "ksh"}


def _tokenize(command):
    """shlex-tokenize the full command; [] if it cannot be parsed at all."""
    for posix in (True, False):
        try:
            return shlex.split(command, posix=posix)
        except ValueError:
            continue
    return []


def _cmd_word(tok):
    """Reduce a token to its command word, stripping subshell / substitution /
    backtick / ``VAR=`` punctuation (so ``(git`` / ``$(git`` / ``x=$(git`` -> ``git``)."""
    t, prev = tok, None
    while t != prev:
        prev = t
        t = t.lstrip("(){}`")
        if t.startswith("$("):
            t = t[2:]
        t = re.sub(r"^[A-Za-z_]\w*=", "", t)
    return t


def _clean(tok):
    """Strip surrounding subshell/backtick punctuation from an argument token."""
    return tok.strip("(){}`")


def _is_cmd(cmd_word, name):
    return cmd_word == name or cmd_word.endswith("/" + name)


def _split_subcommand(gargs):
    """Given the tokens after ``git``, skip global options and return (sub, subargs)."""
    i = 0
    while i < len(gargs):
        t = gargs[i]
        if t in _GLOBAL_WITH_ARG:
            i += 2
            continue
        if t.startswith("-"):
            i += 1
            continue
        return t, gargs[i + 1 :]
    return None, []


def parse_git_invocations(command):
    """Return a list of (subcommand, subargs) for each ``git <sub> ...`` found.

    ``subargs`` tokens are cleaned of wrapping shell punctuation. Quoted strings
    (commit messages, etc.) remain single opaque tokens.
    """
    tokens = _tokenize(command)
    out = []
    i, n = 0, len(tokens)
    while i < n:
        cw = _cmd_word(tokens[i])

        # Recurse into `bash -c "<payload>"` / `sh -c '...'`
        if any(_is_cmd(cw, s) for s in _SHELLS):
            j, bargs = i + 1, []
            while j < n and tokens[j] not in _SEPARATORS:
                bargs.append(tokens[j])
                j += 1
            for k, a in enumerate(bargs):
                if a == "-c" or re.fullmatch(r"-[A-Za-z]*c[A-Za-z]*", a):
                    payload = next((b for b in bargs[k + 1 :] if not b.startswith("-")), None)
                    if payload:
                        out.extend(parse_git_invocations(payload))
                    break
            i = j
            continue

        if _is_cmd(cw, "git"):
            j, gargs = i + 1, []
            while j < n and tokens[j] not in _SEPARATORS:
                gargs.append(_clean(tokens[j]))
                j += 1
            sub, subargs = _split_subcommand(gargs)
            if sub is not None:
                out.append((sub, subargs))
            i = j
            continue

        i += 1
    return out


def has_short_flag(tokens, ch):
    """True if a single-dash short-flag cluster (e.g. -f, -nm) contains `ch`."""
    return any(re.fullmatch(r"-[A-Za-z]+", t) and ch in t[1:] for t in tokens)


def is_main_ref(arg):
    """True if a branch/ref argument refers to the main branch (bare or fully-qualified)."""
    return arg in ("main", "refs/heads/main")


def _git_stdout(cwd, args):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout
    except Exception:
        return ""


def status_paths(cwd):
    """Working-tree paths from ``git status --porcelain -z`` (NUL-safe, rename-aware)."""
    fields = _git_stdout(cwd, ["status", "--porcelain", "-z"]).split("\0")
    paths, i = [], 0
    while i < len(fields):
        entry = fields[i]
        if not entry:
            i += 1
            continue
        xy = entry[:2]
        path = entry[3:] if len(entry) > 3 else ""
        i += 2 if ("R" in xy or "C" in xy) else 1  # rename/copy emits an extra source field
        if path:
            paths.append(path)
    return paths


def diff_paths(cwd, args):
    """Paths from a ``git diff ... --name-only`` invocation (NUL-safe)."""
    raw = _git_stdout(cwd, [*args, "-z"])
    return [p for p in raw.split("\0") if p]
