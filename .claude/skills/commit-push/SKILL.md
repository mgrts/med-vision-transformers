---
name: commit-push
description: Run code-review, byte-compile src/, and the pre-commit / make lint hooks; update docs if drifted; write a Conventional Commits message; commit and push to main on GitHub (origin mgrts/med-vision-transformers); optionally bump the [tool.poetry] version (Poetry) and push a release tag. Stops at every gate (failed review, compile error, failed lint/hooks, conflicting rebase) and requires explicit confirmation before committing and pushing. NEVER adds Claude/AI commit attribution.
---

# Commit & push for med-vision-transformers

Analyze pending changes, review them, compile + lint, update docs if needed, write a
Conventional Commits message, and push to `main`. Optionally bump the package version and
push a release tag.

The default branch is **`main`**; origin is **`git@github.com:mgrts/med-vision-transformers.git`**
(GitHub, owner `mgrts`). This is a solo research repo, so the default flow pushes directly
to `main` after gates pass and the user confirms.

## Arguments

`$ARGUMENTS` — optional. A free-form commit message (used verbatim as the subject after type
inference) and/or flags: `--no-push` (commit only), `--release` (also bump version and offer
a tag). There is **no issue tracker** — never invent ticket references.

## Important

- **Conventional Commits**: `type(scope): subject`. Types: `feat`, `fix`, `refactor`,
  `perf`, `test`, `docs`, `chore`, `build`, `ci`. Scope optional but encouraged (e.g.
  `models`, `train`, `data`, `eval`, `predict`, `config`, `losses`, `mim`, `cv`).
- **NEVER** list Claude among commit authors. Do not add a `Co-Authored-By` trailer, set
  `--author` to Claude/Anthropic, use an `@anthropic.com` address, or add a "Generated with
  Claude" line — to the commit message OR a PR body. This is a hard project rule: the
  `guard_git` PreToolUse hook **blocks** any `git commit` carrying such attribution, so a
  slip is denied rather than committed.
- Do **NOT** use `--force`, `--no-verify`, or any destructive git flag — the `guard_git` hook
  blocks these. If a step fails, stop and ask the user.

## Flow

### Step 1: Gather changes

```bash
git status --short
git diff --staged --stat
git diff --stat
git branch --show-current
```

If there are no changes, stop: "Nothing to commit." If the current branch is not `main`,
note it and ask whether to proceed on this branch or switch.

### Step 2: Run the code-review skill

Invoke the `code-review` skill on the pending diff.

- **Critical / High** findings: stop. Show them and ask whether to proceed anyway, fix
  automatically, or cancel. Do not move on without explicit acknowledgement.
- **Medium / Low** findings: print as a heads-up and continue.

### Step 3: Compile check

```bash
poetry run python -m compileall -q src
```

There is no test suite, so this is the baseline correctness gate. If it fails, show the
error and fix it (or stop and ask). For a change that could plausibly break at runtime
(training/CV/eval logic, dataset construction, model shapes), offer to run the `/verify`
skill (a 1-epoch / 2-fold smoke run) before committing.

### Step 4: Run lint / pre-commit hooks

```bash
make lint            # flake8 + isort --check + black --check (line 99)
```

If it fails: `make format` (isort + black) auto-fixes formatting — run it, then re-run
`make lint`. If `pre-commit` is installed and configured, `poetry run pre-commit run
--all-files` may also be run; never bypass with `--no-verify`. If `check-added-large-files`
or `detect-private-key` trips, do NOT force it through — surface the offending file.

### Step 5: Update documentation

Read `README.md` and `CLAUDE.md`; update only sections that drifted from reality:

- **New module under `src/`** → `CLAUDE.md` package map (+ README structure tree).
- **New `config.py` constant or changed default / hyperparameter** → relevant doc line.
- **A CRITICAL invariant changed** (masking/normalization/binary-head/leakage-split/
  reproducibility/model-signature) → update the matching `CLAUDE.md` invariant.
- **New CLI option or training/eval entrypoint behavior** → README "How to run".

If nothing drifted, skip this step. Do not rewrite docs that are already correct.

### Step 6: Optional version bump + release (only if `--release` or the user asks)

By default, do NOT bump the version on every commit. If a release is requested:

- Patch-bump `version` under **`[tool.poetry]`** in `pyproject.toml` (poetry-core backend),
  e.g. `0.0.1 → 0.0.2`. Never add a PEP 621 `[project]` table. (`poetry version patch` does
  this.)
- Form the tag `v<version>` — created in Step 10 after the push.

### Step 7: Generate the Conventional Commits message

**Subject** (≤ 72 chars): `type(scope): summary`. Infer the type from the diff:

- new capability (loss, dataset, model, task, CLI option) → `feat`
- bug fix → `fix`
- behaviour-preserving restructure → `refactor`
- speed/memory → `perf`
- docs/CLAUDE.md only → `docs`
- tooling/deps/version → `chore` / `build`

If `$ARGUMENTS` supplied a message, use it verbatim as the subject (after the type).

**Body** (after a blank line): one line per significant change. If a CRITICAL invariant or a
cross-file contract changed (masking, normalization, the binary head, the split/CV protocol,
model signatures), explicitly note the synchronized updates so the contract reads as
kept-whole. Add the version line only if Step 6 bumped it:

```
Version: 0.0.1 -> 0.0.2
```

**No AI-attribution trailer** (see Important).

### Step 8: Show summary and confirm

Print: code-review result, compile result, lint result, doc updates (or "none"), version
bump (or "none"), files to be committed (`git status --short`), and the full commit message.
Then ask with `AskUserQuestion`:

```
question: "Commit and push to origin/main?"
header: "Commit & Push"
options:
  - "Yes" — stage all changes, commit, rebase onto origin/main, push.
  - "No"  — cancel, leave the working tree as-is.
```

Do NOT proceed without an explicit "Yes". If `--no-push` was passed, the option is
"Commit only (no push)".

### Step 9: Commit and push

```bash
git add -A
git commit -m "<subject>

<body>"
git fetch origin main
git rebase origin/main
```

If the rebase conflicts, **abort** (`git rebase --abort`) and tell the user to resolve
manually — do not auto-resolve. Then (unless `--no-push`):

```bash
git push origin main
```

If the push fails (branch protection, auth, network), do NOT retry and do NOT force. Show the
error and suggest a feature branch + PR. `gh` may not be installed here; if it is absent,
suggest `git switch -c <branch> && git push -u origin <branch>` and opening the PR on GitHub
in the browser (otherwise `gh pr create`).

### Step 10: Optional release tag

Only if Step 6 bumped the version. Read `version` from `pyproject.toml`, form `v<version>`.
Check it does not already exist:

```bash
git rev-parse "v<version>" 2>/dev/null
```

If it exists, surface that and skip. Otherwise confirm with `AskUserQuestion`, then:

```bash
git tag -a "v<version>" -m "Release v<version>"
git push origin "v<version>"
```

If the tag push fails, do NOT retry or delete the local tag; report that it exists locally.

### Step 11: Final report

```
Pushed to origin/main.
Review: passed (or: N findings)   Compile: ok   Lint: passed
Doc updates: <files or "none">
Version: <bump or "no bump">      Tag: <v.. pushed | skipped>
```

Or, if the push was blocked, show the error and the feature-branch + PR suggestion.
