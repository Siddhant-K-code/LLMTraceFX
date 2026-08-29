#!/usr/bin/env bash
# Tests for scripts/lint-changed.sh.
#
# The ratchet's real job is deciding which files changed. Getting that wrong in
# the quiet direction, reporting "nothing to check" when it cannot actually tell,
# is far worse than getting it wrong loudly: a check that runs on an empty file
# list reports success, so the gate silently stops gating. Several cases below
# exist purely to pin that behaviour down.
#
# The linters themselves are not exercised. A stub `uv` is put on PATH which
# records the arguments it was handed and returns a controlled exit code, so each
# case can assert on the exact file list the script builds and on the status it
# propagates, without needing a real toolchain in a throwaway repository.
#
# Usage: scripts/test-lint-changed.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# Overridable so the suite can be pointed at an older revision of the script to
# confirm these cases actually fail against the behaviour they describe.
SCRIPT="${LINT_CHANGED_SCRIPT:-${SCRIPT_DIR}/lint-changed.sh}"
EMPTY_TREE="4b825dc642cb6eb9a060e54bf8d69288fbee4904"

PASSED=0
FAILED=0
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

pass() {
    printf '  ok    %s\n' "$1"
    PASSED=$((PASSED + 1))
}

fail() {
    printf '  FAIL  %s\n' "$1"
    printf '        expected: %s\n' "$2"
    printf '        actual:   %s\n' "$3"
    FAILED=$((FAILED + 1))
}

check() {
    if [ "$2" = "$3" ]; then
        pass "$1"
    else
        fail "$1" "$2" "$3"
    fi
}

# A throwaway repository with a stub `uv` ahead of the real one on PATH.
new_repo() {
    REPO="$(mktemp -d "${WORKDIR}/repo.XXXXXX")"
    STUB_LOG="${REPO}/stub.log"
    mkdir -p "${REPO}/bin"
    cat >"${REPO}/bin/uv" <<'STUB'
#!/usr/bin/env bash
{
    printf 'CALL'
    for arg in "$@"; do printf '\t%s' "$arg"; done
    printf '\n'
} >>"$STUB_LOG"
exit "${STUB_EXIT:-0}"
STUB
    chmod +x "${REPO}/bin/uv"
    : >"$STUB_LOG"
    git -C "$REPO" init -q
    git -C "$REPO" config user.email test@example.com
    git -C "$REPO" config user.name "Test"
    git -C "$REPO" config commit.gpgsign false
}

commit_file() {
    local path="$1"
    local body="${2:-x = 1}"
    local dir
    dir="$(dirname "$path")"
    [ "$dir" = "." ] || mkdir -p "${REPO}/${dir}"
    printf '%s\n' "$body" >"${REPO}/${path}"
    git -C "$REPO" add -- "$path"
    git -C "$REPO" commit -q -m "touch ${path}"
}

# Runs the script under test inside $REPO. Sets RC and OUT.
run_script() {
    ( cd "$REPO" && PATH="${REPO}/bin:${PATH}" STUB_LOG="$STUB_LOG" \
        STUB_EXIT="${STUB_EXIT:-0}" "$SCRIPT" "$@" ) >"${REPO}/out.txt" 2>&1
    RC=$?
    OUT="$(cat "${REPO}/out.txt")"
}

# The files handed to a given tool, sorted, comma separated. Arguments are
# recorded tab separated, so a path containing spaces stays a single field and a
# split-argument bug shows up as two entries rather than being masked.
files_for() {
    local tool="$1"
    grep -F "$(printf '\t%s\t' "$tool")" "$STUB_LOG" 2>/dev/null | head -1 \
        | tr '\t' '\n' | awk 'seen { print } $0 == "--" { seen = 1 }' \
        | LC_ALL=C sort | paste -sd, -
}

calls_for() {
    grep -cF "$(printf '\t%s\t' "$1")" "$STUB_LOG" 2>/dev/null | tr -d ' '
}

echo "Testing ${SCRIPT}"
echo

# --- file selection -------------------------------------------------------

new_repo
commit_file "a.py"
BASE="$(git -C "$REPO" rev-parse HEAD)"
printf 'docs\n' >"${REPO}/README.md"
git -C "$REPO" add -- README.md
git -C "$REPO" commit -q -m "docs only"
run_script "$BASE"
check "no Python changes exits 0" "0" "$RC"
check "no Python changes runs no tools" "0" "$(calls_for ruff)"

new_repo
commit_file "a.py"
BASE="$(git -C "$REPO" rev-parse HEAD)"
commit_file "b.py"
run_script "$BASE"
check "added file is checked" "b.py" "$(files_for ruff)"

new_repo
commit_file "a.py"
BASE="$(git -C "$REPO" rev-parse HEAD)"
commit_file "a file with spaces.py"
run_script "$BASE"
check "path with spaces stays one argument" "a file with spaces.py" "$(files_for ruff)"

new_repo
commit_file "a.py"
commit_file "b.py"
BASE="$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" rm -q -- b.py
git -C "$REPO" commit -q -m "delete b"
printf 'x = 2\n' >"${REPO}/a.py"
git -C "$REPO" add -- a.py
git -C "$REPO" commit -q -m "edit a"
run_script "$BASE"
check "deleted file is not linted" "a.py" "$(files_for ruff)"

new_repo
commit_file "old.py"
BASE="$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" mv old.py new.py
git -C "$REPO" commit -q -m "rename"
run_script "$BASE"
check "rename reports the new path" "new.py" "$(files_for ruff)"

# A force push moves the branch by more than one commit. Falling back to HEAD^
# would only ever see the last of these three files.
new_repo
commit_file "a.py"
BASE="$(git -C "$REPO" rev-parse HEAD)"
commit_file "one.py"
commit_file "two.py"
commit_file "three.py"
run_script "$BASE"
check "multi commit push checks every commit" "one.py,three.py,two.py" "$(files_for ruff)"

# --- failing closed -------------------------------------------------------

new_repo
commit_file "a.py"
MAIN="$(git -C "$REPO" rev-parse HEAD)"
git -C "$REPO" checkout -q --orphan unrelated
git -C "$REPO" rm -rq --cached .
rm -f "${REPO}/a.py"
commit_file "z.py"
git -C "$REPO" checkout -q -f "$MAIN"
UNRELATED="$(git -C "$REPO" rev-parse unrelated)"
run_script "$UNRELATED"
check "unrelated history fails closed" "1" "$RC"
check "unrelated history runs no tools" "0" "$(calls_for ruff)"

new_repo
commit_file "a.py"
run_script "0000000000000000000000000000000000000000"
check "invalid base fails closed" "1" "$RC"
check "invalid base runs no tools" "0" "$(calls_for ruff)"

new_repo
commit_file "a.py"
run_script "refs/heads/does-not-exist"
check "missing base ref fails closed" "1" "$RC"

# --- initial push ---------------------------------------------------------

# A root commit has no parent, so there is nothing to diff against. The empty
# tree is the honest baseline and yields every tracked file.
new_repo
commit_file "a.py"
commit_file "pkg/b.py"
run_script "$EMPTY_TREE"
check "empty tree base checks every tracked file" "a.py,pkg/b.py" "$(files_for ruff)"
check "empty tree base exits 0 when clean" "0" "$RC"

# --- status propagation ---------------------------------------------------

new_repo
commit_file "a.py"
BASE="$(git -C "$REPO" rev-parse HEAD)"
commit_file "b.py"
STUB_EXIT=1 run_script "$BASE"
check "a failing tool fails the script" "1" "$RC"
unset STUB_EXIT

# --- mypy scoping ---------------------------------------------------------

new_repo
commit_file "llmtracefx/__init__.py"
BASE="$(git -C "$REPO" rev-parse HEAD)"
commit_file "llmtracefx/mod.py"
commit_file "tests/test_mod.py"
run_script "$BASE"
check "ruff sees package and tests" "llmtracefx/mod.py,tests/test_mod.py" "$(files_for ruff)"
check "mypy sees only the package" "llmtracefx/mod.py" "$(files_for mypy)"

new_repo
commit_file "a.py"
BASE="$(git -C "$REPO" rev-parse HEAD)"
commit_file "tests/test_only.py"
run_script "$BASE"
check "mypy is skipped with no package files" "0" "$(calls_for mypy)"

echo
echo "passed ${PASSED}, failed ${FAILED}"
[ "$FAILED" -eq 0 ]
