#!/usr/bin/env bash
#
# Quality ratchet: run the linters over only the Python files that changed
# relative to a base commit, instead of over the whole repository.
#
# Why: the repository carries a large amount of pre-existing lint, format and
# type debt on main. A repo-wide gate would be red on every pull request and
# would quickly be ignored. Checking only changed files keeps new code at the
# intended standard without demanding a mass reformat first.
#
# The tools are ruff (linting only), black, isort and mypy. `ruff format` is
# deliberately not used: it and black disagree on a few files here and each
# reverts the other, so running both can never pass. black is the formatter of
# record. ruff's linter and black do not conflict, and ruff's I001 import rule
# was verified to agree with isort, so `ruff check` stays.
#
# Usage:
#   scripts/lint-changed.sh [BASE]
#
# BASE defaults to origin/main. CI passes an explicit base SHA.
#
# Notes on the file selection, which is deliberately picky:
#   merge base     resolved explicitly, then used as a two dot diff. Equivalent
#                  to --merge-base, but a missing merge base can be reported on
#                  its own rather than surfacing as a generic diff error.
#   --diff-filter   A/C/M/R only. Deleted files are excluded so no tool is ever
#                   handed a path that no longer exists. For a rename, git
#                   reports the new path, which is the one worth checking.
#   -z              NUL separated output, so paths containing spaces survive
#   -- '*.py'       a git pathspec, which matches nested directories
#
# Every failure here is a hard failure. A ratchet that cannot work out what
# changed must not fall back to checking nothing, because that reports success.
# scripts/test-lint-changed.sh covers the cases that used to slip through.
#
# Written for bash 3.2 so it also runs on a stock macOS shell. That rules out
# mapfile, hence the read -d '' loop below.

set -euo pipefail

BASE="${1-origin/main}"

# `${1-...}` rather than `${1:-...}` on purpose. An explicitly passed empty
# string means the caller tried to supply a base and came up with nothing, which
# is the "cannot work out what changed" condition, not a request for the default.
# Coercing it to origin/main would diff HEAD against itself on a push build and
# pass having checked nothing.
if [ "$#" -gt 0 ] && [ -z "$1" ]; then
    echo "lint-changed: empty base argument." >&2
    echo "lint-changed: pass a base ref or SHA, or pass nothing for origin/main." >&2
    exit 1
fi

# The empty tree. Used as the base for an initial push, where there is no
# previous commit to compare against and every tracked file is new.
EMPTY_TREE="4b825dc642cb6eb9a060e54bf8d69288fbee4904"

if [ "$BASE" = "$EMPTY_TREE" ]; then
    # No merge base exists against a bare tree, and none is wanted. Comparing
    # HEAD to the empty tree yields every tracked Python file, which is the
    # correct baseline for a branch with no history behind it.
    DIFF_BASE="$EMPTY_TREE"
else
    if ! git rev-parse --verify --quiet "${BASE}^{commit}" >/dev/null; then
        echo "lint-changed: base commit '${BASE}' not found." >&2
        echo "lint-changed: fetch the base branch, or pass an explicit base SHA." >&2
        exit 1
    fi

    # Resolving the merge base as its own step means an unrelated history gets a
    # clear message. Folding it into `git diff --merge-base` would still fail,
    # but the reason would be buried in a diff error.
    if ! DIFF_BASE="$(git merge-base "$BASE" HEAD 2>/dev/null)"; then
        echo "lint-changed: no merge base between '${BASE}' and HEAD." >&2
        echo "lint-changed: refusing to guess a base, because checking nothing" >&2
        echo "lint-changed: would silently pass the ratchet." >&2
        exit 1
    fi
fi

# The file list goes through a temporary file so the exit status of git diff can
# be checked. Read through a process substitution instead, a git failure is
# invisible: the loop reads nothing, the list comes out empty, and the script
# reports "nothing to check" and exits 0. That turns any git error into a silent
# pass, which is the one failure mode a ratchet must not have.
DIFF_LIST="$(mktemp)"
trap 'rm -f "$DIFF_LIST"' EXIT

if ! git diff --name-only --diff-filter=ACMR -z "$DIFF_BASE" HEAD -- '*.py' >"$DIFF_LIST"; then
    echo "lint-changed: git diff against '${DIFF_BASE}' failed." >&2
    exit 1
fi

CHANGED=()
while IFS= read -r -d '' file; do
    CHANGED+=("$file")
done <"$DIFF_LIST"

if [ ${#CHANGED[@]} -eq 0 ]; then
    echo "No Python files changed against ${BASE}. Nothing to check."
    exit 0
fi

echo "Checking ${#CHANGED[@]} changed Python file(s) against ${BASE}:"
for file in "${CHANGED[@]}"; do
    echo "  ${file}"
done
echo

# mypy is scoped to the package only. The strict settings in pyproject.toml
# (disallow_untyped_defs) flag ordinary untyped test functions, and the existing
# `make lint` target already scopes mypy to llmtracefx/.
MYPY_FILES=()
for file in "${CHANGED[@]}"; do
    case "$file" in
        llmtracefx/*) MYPY_FILES+=("$file") ;;
    esac
done

STATUS=0

run_check() {
    label="$1"
    shift
    echo "--- ${label} ---"
    if "$@"; then
        echo "${label}: ok"
    else
        echo "${label}: FAILED"
        STATUS=1
    fi
    echo
}

run_check "ruff check" uv run ruff check -- "${CHANGED[@]}"
run_check "black" uv run black --check -- "${CHANGED[@]}"
run_check "isort" uv run isort --check-only -- "${CHANGED[@]}"

if [ ${#MYPY_FILES[@]} -gt 0 ]; then
    # --follow-imports=silent stops mypy reporting errors that live in unchanged
    # modules pulled in by an import. Without it, editing a clean file can fail
    # the build because of type debt somewhere else entirely.
    run_check "mypy" uv run mypy --follow-imports=silent -- "${MYPY_FILES[@]}"
else
    echo "--- mypy ---"
    echo "mypy: skipped, no changed files under llmtracefx/"
    echo
fi

if [ "$STATUS" -ne 0 ]; then
    echo "Quality ratchet failed. The issues above are in files this change touches."
    exit 1
fi

echo "Quality ratchet passed."
