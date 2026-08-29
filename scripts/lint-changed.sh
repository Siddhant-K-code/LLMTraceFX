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
#   --merge-base    three-dot semantics, so unrelated commits that landed on the
#                   base branch are not attributed to this change
#   --diff-filter   A/C/M/R only. Deleted files are excluded so no tool is ever
#                   handed a path that no longer exists. For a rename, git
#                   reports the new path, which is the one worth checking.
#   -z              NUL separated output, so paths containing spaces survive
#   -- '*.py'       a git pathspec, which matches nested directories
#
# Written for bash 3.2 so it also runs on a stock macOS shell. That rules out
# mapfile, hence the read -d '' loop below.

set -euo pipefail

BASE="${1:-origin/main}"

if ! git rev-parse --verify --quiet "${BASE}^{commit}" >/dev/null; then
    echo "lint-changed: base commit '${BASE}' not found." >&2
    echo "lint-changed: fetch the base branch, or pass an explicit base SHA." >&2
    exit 1
fi

CHANGED=()
while IFS= read -r -d '' file; do
    CHANGED+=("$file")
done < <(git diff --merge-base "$BASE" HEAD --name-only --diff-filter=ACMR -z -- '*.py')

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
