#!/bin/sh

SAFE_PATH=/usr/bin:/bin:/usr/local/bin

# Sanitize before invoking stat, shasum, sed, or any other external utility.
# The marker is consumed immediately and is never passed to Python.
if [ "${LLMTRACEFX_CLEAN_ENV_BOOTSTRAPPED-}" != "1" ]; then
    exec /usr/bin/env -i \
        PATH="$SAFE_PATH" LANG=C LC_ALL=C \
        LLMTRACEFX_CLEAN_ENV_BOOTSTRAPPED=1 \
        /bin/sh "$0" "$@"
fi
unset LLMTRACEFX_CLEAN_ENV_BOOTSTRAPPED

set -eu
set -f
umask 077

PROGRAM=${0##*/}

fail() {
    printf '%s\n' "$PROGRAM: error: $1" >&2
    exit 2
}

usage() {
    printf '%s\n' \
        "usage: $PROGRAM preflight --cli ABSOLUTE_CLI --cli-sha256 SHA256" \
        "       $PROGRAM run --cli ABSOLUTE_CLI --cli-sha256 SHA256 --execution-config ABSOLUTE_CONFIG --authorization ABSOLUTE_AUTHORIZATION --output-dir ABSOLUTE_OUTPUT_DIR" >&2
    exit 2
}

require_unambiguous_absolute_path() {
    label=$1
    path=$2
    case "$path" in
        /*) ;;
        *) fail "$label must be an absolute path" ;;
    esac
    case "$path" in
        *//* | */./* | */../* | */. | */..) fail "$label path is ambiguous" ;;
    esac
}

require_no_symlink_components() {
    label=$1
    path=$2
    cursor=$path
    while [ "$cursor" != "/" ]; do
        [ ! -L "$cursor" ] ||
            fail "$label must not contain symlink components"
        parent=${cursor%/*}
        [ -n "$parent" ] || parent=/
        cursor=$parent
    done
}

file_uid() {
    if /usr/bin/stat -f '%u' / >/dev/null 2>&1; then
        /usr/bin/stat -f '%u' "$1"
    else
        /usr/bin/stat -c '%u' "$1"
    fi
}

file_mode() {
    if /usr/bin/stat -f '%Lp' / >/dev/null 2>&1; then
        /usr/bin/stat -f '%Lp' "$1"
    else
        /usr/bin/stat -c '%a' "$1"
    fi
}

require_trusted_directory_chain() {
    label=$1
    directory=$2
    current_uid=$(/usr/bin/id -u) ||
        fail "current user could not be identified"
    cursor=$directory
    while :; do
        [ -d "$cursor" ] && [ ! -L "$cursor" ] ||
            fail "$label has an unsafe directory component"
        owner=$(file_uid "$cursor") ||
            fail "$label directory owner could not be inspected"
        [ "$owner" = "$current_uid" ] || [ "$owner" = "0" ] ||
            fail "$label directory must be owned by the current user or root"
        mode=$(file_mode "$cursor") ||
            fail "$label directory mode could not be inspected"
        if [ "$((0$mode & 022))" -ne 0 ]; then
            [ "$owner" = "0" ] && [ "$((0$mode & 01000))" -ne 0 ] ||
                fail "$label directory must not be group- or world-writable"
        fi
        [ "$cursor" = "/" ] && break
        parent=${cursor%/*}
        [ -n "$parent" ] || parent=/
        cursor=$parent
    done
}

require_current_owner() {
    label=$1
    path=$2
    owner=$(file_uid "$path") || fail "$label owner could not be inspected"
    current=$(/usr/bin/id -u) || fail "current user could not be identified"
    [ "$owner" = "$current" ] || fail "$label must be owned by the current user"
}

require_regular_nonsymlink_file() {
    label=$1
    path=$2
    require_unambiguous_absolute_path "$label" "$path"
    require_no_symlink_components "$label" "$path"
    [ -f "$path" ] || fail "$label must be a regular file"
    [ ! -L "$path" ] || fail "$label must not be a symlink"
    require_trusted_directory_chain "$label" "${path%/*}"
    require_current_owner "$label" "$path"
}

require_private_file() {
    label=$1
    path=$2
    require_regular_nonsymlink_file "$label" "$path"
    mode=$(file_mode "$path") || fail "$label mode could not be inspected"
    [ "$mode" = "600" ] || fail "$label must have mode 0600"
}

require_output_directory() {
    path=$1
    require_unambiguous_absolute_path "output directory" "$path"
    require_no_symlink_components "output directory" "$path"
    [ -d "$path" ] || fail "output directory must already exist"
    [ ! -L "$path" ] || fail "output directory must not be a symlink"
    require_trusted_directory_chain "output directory" "$path"
    require_current_owner "output directory" "$path"
    mode=$(file_mode "$path") || fail "output directory mode could not be inspected"
    [ "$mode" = "700" ] || fail "output directory must have mode 0700"
    [ -z "$(/bin/ls -A "$path")" ] || fail "output directory must be empty"
}

sha256_file() {
    path=$1
    if [ -x /usr/bin/sha256sum ]; then
        digest=$(/usr/bin/sha256sum "$path") ||
            fail "CLI SHA-256 could not be calculated"
    elif [ -x /bin/sha256sum ]; then
        digest=$(/bin/sha256sum "$path") ||
            fail "CLI SHA-256 could not be calculated"
    elif [ -x /usr/bin/shasum ]; then
        digest=$(/usr/bin/shasum -a 256 "$path") ||
            fail "CLI SHA-256 could not be calculated"
    else
        fail "no trusted SHA-256 utility is available"
    fi
    printf '%s\n' "${digest%% *}"
}

extract_cli_interpreter() {
    cli=$1
    first_line=$(/usr/bin/sed -n '1p' "$cli") ||
        fail "CLI executable shebang could not be inspected"
    case "$first_line" in
        '#!/bin/sh')
            second_line=$(/usr/bin/sed -n '2p' "$cli") ||
                fail "CLI executable trampoline could not be inspected"
            prefix="'''exec' "
            suffix=' "$0" "$@"'
            case "$second_line" in
                "$prefix"*"$suffix")
                    candidate=${second_line#"$prefix"}
                    candidate=${candidate%"$suffix"}
                    case "$candidate" in
                        \'*\')
                            candidate=${candidate#\'}
                            candidate=${candidate%\'}
                            case "$candidate" in
                                *\'*) fail "CLI executable trampoline is malformed" ;;
                            esac
                            ;;
                    esac
                    ;;
                *) fail "CLI executable trampoline is malformed" ;;
            esac
            ;;
        '#!'/*)
            candidate=${first_line#\#!}
            case "$candidate" in
                *' '* | *'	'*)
                    fail "CLI executable shebang must not contain arguments"
                    ;;
            esac
            ;;
        *)
            fail "CLI executable must use an absolute Python shebang or wheel trampoline"
            ;;
    esac
    case "$candidate" in
        *'	'*) fail "CLI interpreter path must not contain tabs" ;;
    esac
    require_unambiguous_absolute_path "CLI interpreter" "$candidate"
    printf '%s\n' "$candidate"
}

resolve_cli_interpreter() {
    interpreter=$1
    links=0
    while :; do
        interpreter_directory=$(
            CDPATH= cd -P "${interpreter%/*}" 2>/dev/null && /bin/pwd -P
        ) || fail "CLI interpreter directory could not be resolved"
        require_unambiguous_absolute_path \
            "CLI interpreter directory" "$interpreter_directory"
        require_no_symlink_components \
            "CLI interpreter directory" "$interpreter_directory"
        interpreter="$interpreter_directory/${interpreter##*/}"
        [ -L "$interpreter" ] || break
        require_current_owner "CLI interpreter link" "$interpreter"
        target=$(/usr/bin/readlink "$interpreter") ||
            fail "CLI interpreter link could not be inspected"
        case "$target" in
            /*) interpreter=$target ;;
            */*) fail "CLI interpreter relative link must be a simple file name" ;;
            *) interpreter=${interpreter%/*}/$target ;;
        esac
        require_unambiguous_absolute_path "CLI interpreter" "$interpreter"
        links=$((links + 1))
        [ "$links" -le 16 ] || fail "CLI interpreter has too many symlink levels"
    done
    require_no_symlink_components "CLI interpreter" "$interpreter"
    [ -f "$interpreter" ] && [ -x "$interpreter" ] ||
        fail "CLI interpreter is not an executable regular file"
    require_trusted_directory_chain "CLI interpreter" "${interpreter%/*}"
    owner=$(file_uid "$interpreter") ||
        fail "CLI interpreter owner could not be inspected"
    current=$(/usr/bin/id -u) || fail "current user could not be identified"
    [ "$owner" = "$current" ] || [ "$owner" = "0" ] ||
        fail "CLI interpreter must be owned by the current user or root"
    mode=$(file_mode "$interpreter") ||
        fail "CLI interpreter mode could not be inspected"
    [ "$((0$mode & 022))" -eq 0 ] ||
        fail "CLI interpreter must not be group- or world-writable"
    printf '%s\n' "$interpreter"
}

require_cli() {
    cli=$1
    expected_sha256=$2
    require_regular_nonsymlink_file "CLI executable" "$cli"
    [ "${cli##*/}" = "llmtracefx-vllm-kv-truth" ] ||
        fail "CLI executable has the wrong installed name"
    [ -x "$cli" ] || fail "CLI executable is not executable"
    mode=$(file_mode "$cli") || fail "CLI executable mode could not be inspected"
    [ "$((0$mode & 022))" -eq 0 ] ||
        fail "CLI executable must not be group- or world-writable"
    [ "${#expected_sha256}" -eq 64 ] ||
        fail "CLI SHA-256 must be exactly 64 lowercase hexadecimal characters"
    case "$expected_sha256" in
        *[!0-9a-f]*)
            fail "CLI SHA-256 must be exactly 64 lowercase hexadecimal characters"
            ;;
    esac
    actual_sha256=$(sha256_file "$cli")
    [ "$actual_sha256" = "$expected_sha256" ] ||
        fail "CLI executable does not match the expected SHA-256"

    interpreter=$(extract_cli_interpreter "$cli")
    case "${interpreter##*/}" in
        python | python3 | python3.[0-9] | python3.[0-9][0-9]) ;;
        *) fail "CLI executable shebang must name a supported Python interpreter" ;;
    esac
    cli_dir=${cli%/*}
    interpreter_dir=${interpreter%/*}
    [ "$interpreter_dir" = "$cli_dir" ] ||
        fail "CLI executable must use the interpreter from its installation directory"
    original_interpreter=$interpreter
    resolve_cli_interpreter "$interpreter" >/dev/null
    interpreter=$original_interpreter
}

require_unambiguous_absolute_path "launcher" "$0"
require_no_symlink_components "launcher" "$0"
[ -f "$0" ] && [ ! -L "$0" ] || fail "launcher must be a regular non-symlink file"
require_current_owner "launcher" "$0"
launcher_mode=$(file_mode "$0") || fail "launcher mode could not be inspected"
[ "$((0$launcher_mode & 022))" -eq 0 ] ||
    fail "launcher must not be group- or world-writable"

[ "$#" -ge 1 ] || usage
command=$1

case "$command" in
    preflight)
        [ "$#" -eq 5 ] || usage
        [ "$2" = "--cli" ] && [ "$4" = "--cli-sha256" ] || usage
        cli=$3
        cli_sha256=$5
        require_cli "$cli" "$cli_sha256"
        exec /usr/bin/env -i \
            PATH="$SAFE_PATH" LANG=C LC_ALL=C \
            "$interpreter" -I "$cli" preflight-clean-environment
        ;;
    run)
        [ "$#" -eq 11 ] || usage
        [ "$2" = "--cli" ] &&
            [ "$4" = "--cli-sha256" ] &&
            [ "$6" = "--execution-config" ] &&
            [ "$8" = "--authorization" ] &&
            [ "${10}" = "--output-dir" ] || usage
        cli=$3
        cli_sha256=$5
        execution_config=$7
        authorization=$9
        output_dir=${11}
        require_cli "$cli" "$cli_sha256"
        require_private_file "execution config" "$execution_config"
        require_private_file "authorization" "$authorization"
        require_output_directory "$output_dir"
        exec /usr/bin/env -i \
            PATH="$SAFE_PATH" LANG=C LC_ALL=C \
            "$interpreter" -I "$cli" run \
            --execution-config "$execution_config" \
            --authorization "$authorization" \
            --output-dir "$output_dir"
        ;;
    *)
        usage
        ;;
esac
