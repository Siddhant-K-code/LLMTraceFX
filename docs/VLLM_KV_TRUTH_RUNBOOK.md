# vLLM KV-cache truth runbook

This runbook executes `qwen3-8b-vllm-kv-truth-v1` only after the implementation
PR is merged and the exact merged head has passed every offline gate. It does
not authorize provider access by itself.

## 1. Prepare the exact merged source archive

From a clean checkout whose `HEAD` is the merged `origin/main`:

```bash
set -euo pipefail
git fetch origin main
test "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)"
test -z "$(git status --porcelain --untracked-files=no)"

export KV_TRUTH_HEAD
KV_TRUTH_HEAD="$(git rev-parse HEAD)"
export KV_TRUTH_ARCHIVE="$PWD/vllm-kv-truth-${KV_TRUTH_HEAD}.tar"
git archive --format=tar --output="$KV_TRUTH_ARCHIVE" "$KV_TRUTH_HEAD"
python3 - <<'PY'
import io
import os
import tarfile

archive = os.environ["KV_TRUTH_ARCHIVE"]
data = (os.environ["KV_TRUTH_HEAD"] + "\n").encode("ascii")
with tarfile.open(archive, "a") as tar:
    marker = tarfile.TarInfo("COMMIT_HEAD")
    marker.size = len(data)
    marker.mode = 0o644
    marker.mtime = 0
    marker.uid = marker.gid = 0
    marker.uname = marker.gname = ""
    tar.addfile(marker, io.BytesIO(data))
PY
export KV_TRUTH_ARCHIVE_SHA256
KV_TRUTH_ARCHIVE_SHA256="$(python3 -c \
  'import hashlib,os; print(hashlib.sha256(open(os.environ["KV_TRUTH_ARCHIVE"],"rb").read()).hexdigest())')"
```

Do not rebuild or replace this archive after authorization is signed.

## 2. Provision and establish temporary key-only access

Provision exactly one RTX 4090 host only after the PR is merged. Generate a
new temporary SSH key with the exact comment
`llmtracefx-kv-truth-authorized-key` (or another safe, unique marker recorded
as `authorized_key_marker` below) and a dedicated known-hosts file. Install
that one public-key line on the host. The orchestrator verifies there is
exactly one authorized-key line ending in the marker and atomically removes
it during teardown. Verify the private key mode is `0600`; never reuse a key
from an earlier VM.

Before GO, perform only the coordinator-approved read-only preflight: boot
time, UTC clock, one GPU, GPU name/memory/driver/compute capability, zero GPU
processes, zero containers, zero swap, RAM, disk, Python 3.12, Docker, and
noninteractive sudo. Do not download the model or image during preflight.

## 3. Write protected execution config

Write a `0600` JSON file with exactly these keys:

```json
{
  "host": "provider-supplied-host-or-ip",
  "port": 22,
  "user": "provider-supplied-user",
  "private_key_path": "/protected/path/to/new-key",
  "known_hosts_path": "/protected/path/to/dedicated-known-hosts",
  "remote_workspace": "/home/provider-user/llmtracefx-kv-truth",
  "authorized_key_marker": "llmtracefx-kv-truth-authorized-key",
  "local_evidence_dir": "/protected/path/to/evidence-output",
  "local_runner_archive": "/absolute/path/to/the-checked-source.tar"
}
```

The SSH port belongs only in this protected file. Never place the target,
user, port, key path, or known-hosts path in shell arguments, logs, evidence,
or the authorization.

## 4. Calculate and sign authorization

The authorization must bind:

- the exact merged repository head and `sha256:<archive digest>`;
- protocol, digest-pinned base image, vLLM commit/version, immutable model
  revision and committed inventory digest;
- one RTX 4090 and exact driver/memory expectations;
- boot/billing timestamp, list rate, total cap, explicit operational cutoff,
  cleanup reserve of at least 35 minutes, authorization time/expiry, nonce;
- zero automatic retries and no replacement.

Compute the catastrophic cap as:

```text
boot + floor_to_microseconds(total_cap / hourly_rate * 3600 seconds)
```

Require both:

```text
now + all remaining stage allowances <= operational cutoff + cleanup reserve
operational cutoff + cleanup reserve <= boot-derived absolute cap
```

Set `authorization_expiry` no earlier than the end of cleanup. Calculate
`authorization_sha256` with
`vllm_kv_truth.lifecycle.build_authorization_seal` over the
authorization object without `authorization_sha256`. Optional detached
OpenSSH signing uses both `signature_path` and `authorized_signers_path`;
supplying only one is invalid.

## 5. Coordinator GO and one command

Only after the coordinator confirms the exact tested merged head,
authorization, remaining reserve, and fresh temporary key, execute:

```bash
llmtracefx-vllm-kv-truth run \
  --execution-config /protected/path/execution-config.json \
  --authorization /protected/path/run-authorization.json
```

There are zero retries and zero replacement runs. The command performs its
own read-only preflight, exact model acquisition and verification, pinned
image pull/build/inspection, event canary, four AB/BA/BA/AB lifecycle pairs,
eviction lane, evidence transfer, local verification/redaction, scoped
cleanup, and OS shutdown.

## 6. Evidence and termination

Keep the transferred private bundle private. Verify the public-redacted
bundle offline:

```bash
llmtracefx-vllm-kv-truth verify-public-bundle \
  --bundle-dir /protected/path/to/evidence-output/public
```

Wait for the exact message:

```text
SAFE TO TERMINATE INSTANCE NOW
```

That message proves only host cleanup checks and shutdown issuance. The
coordinator must separately terminate the reservation in the provider
console and preserve provider confirmation. Never claim provider deletion
from OS state or from the application list-rate ledger.
