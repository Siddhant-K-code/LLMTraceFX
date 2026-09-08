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
The vanilla host does **not** need `huggingface-cli`, `hf`, `pip`, or `uv`.
Model acquisition is deliberately unavailable until the derived image has
passed its downloader attestation.

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
- authorization schema `2` and the exact downloader package
  `huggingface-hub==1.13.0`, interface
  `huggingface_hub.snapshot_download`, and digest-pinned base-image source;
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

The fixed 210-minute reserve ledger is evaluated by stage identity in this
order:

| Stage | Allowance | Reserve at start |
|---|---:|---:|
| Preflight/planning reserve | 15 min | 210 min |
| SSH identity and checked source transfer | 5 min | 195 min |
| Image pull/build/inspect/downloader attestation | 20 min | 190 min |
| In-image model acquisition and host inventory verification | 40 min | 170 min |
| Event-bearing no-warmup canary | 15 min | 130 min |
| Four fresh A/B lifecycle pairs | 60 min | 115 min |
| Fixed eviction lane | 10 min | 55 min |
| Private verify/redact/report/export | 10 min | 45 min |
| Transfer verification, cleanup, key removal, shutdown | 35 min | 35 min |

Reordering image preparation ahead of acquisition does not reduce or
reallocate any allowance: the total remains 210 minutes, the stop-new-work
point remains 175 minutes, and teardown retains its full 35-minute reserve.

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
own read-only preflight, SSH identity and checked source transfer, pinned image
pull/build/inspection and downloader attestation, exact in-image model
acquisition and host-side verification, event canary, four AB/BA/BA/AB
lifecycle pairs, eviction lane, evidence transfer, local
verification/redaction, scoped cleanup, temporary-key removal, and OS
shutdown.

The model-download container is bound to the inspected derived image ID,
labeled with the run nonce, and mounts only the run-scoped model destination
and HF scratch directory. It is the only runtime container with networking
enabled (`bridge`). Its checked Python module calls `snapshot_download` with
the exact model ID and revision, an exact 15-file allowlist, and `token=False`.
Scratch and Hugging Face local metadata are removed before a host-side
15-file/16,397,461,266-byte SHA-256 verification. No GPU container can start
until that verification succeeds. Canary, pair, and eviction containers all
use `--network none` plus the fixed offline environment.

Before the one command above, confirm only this checklist:

1. The source archive is from the exact clean, tested, merged `origin/main`.
2. Authorization schema 2 seals that head, archive digest, runtime/model/image
   pins, downloader identity, budget, nonce, and zero-retry policy.
3. The protected config names a fresh key, dedicated known-hosts file, and
   empty local evidence destination.
4. The coordinator has issued GO and the full 210-minute reserve gate passes.

## 6. Private failure diagnostics

Each command operation appends a mode-`0600`
`private-operation-receipts.jsonl` record under the protected local evidence
directory. This file is private and is not part of the public evidence schema.
Each schema-1 record contains only:

```text
stage, substage, command_description, return_code, timed_out,
stderr_category, stderr_message, reason_code, reserved_minutes
```

No argv, host, user, IP, key/known-hosts path, credential, remote/model path,
prompt/token content, or raw stderr is retained. Failure text is selected from
a bounded allowlist. Current reason codes are:

```text
operation_timeout, operation_start_failed,
preflight_missing_linux, preflight_missing_nvidia_smi,
preflight_missing_docker, preflight_missing_sudo,
preflight_missing_python3, preflight_probe_failed,
identity_gate_failed, source_transfer_failed, image_preparation_failed,
model_download_interface_missing, model_download_version_mismatch,
model_download_failed, model_inventory_mismatch, canary_failed,
pair_lane_failed, eviction_lane_failed, evidence_archive_failed,
evidence_digest_failed, evidence_download_failed,
run_interrupted, teardown_cleanup_failed, teardown_shutdown_failed
```

Successful acquisition also writes a private schema-1
`private-model-acquisition-receipt.json` binding the authorization, inspected
image ID, explicit network mode, run label, downloader package/version/
interface/source, exact model revision, and verified inventory totals.

The terminal reports `stage/substage`, reason code, and the corresponding safe
message. Once trusted configuration and authorization have been loaded and
lifecycle execution begins, teardown still runs after any stage failure or
SIGTERM/SIGHUP. Input or signature rejection happens before all remote
activity and therefore before lifecycle teardown. `SAFE TO TERMINATE INSTANCE
NOW` retains its existing meaning and is emitted only after scoped cleanup,
temporary-key removal, zero-residual checks, and shutdown issuance succeed.
Shutdown is scheduled from the same authenticated SSH session that removes
the temporary key, so key removal cannot prevent the shutdown command from
being issued.

## 7. Evidence and termination

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

## 8. Failed-attempt provenance

The authorized attempt at repository head
`2720134ca6f285d06ae2b42f4fc3d260bda3c45d` stopped in preflight because the
old probe emitted no absolute `HF_CLI` path and the local verifier required
one. The vanilla CloudRift Ubuntu 24.04 host did not provide that executable,
so the marker set was rejected. Teardown succeeded. No model download, image
pull/build, canary, pair, eviction, or GPU workload occurred, so that attempt
created no scientific evidence claim.
