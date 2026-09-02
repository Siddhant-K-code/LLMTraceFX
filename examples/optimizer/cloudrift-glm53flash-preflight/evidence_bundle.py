"""Render and verify the public CloudRift zero-spend refusal bundle."""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

from llmtracefx.deploy.cloudrift import build_cloudrift_plan
from llmtracefx.deploy.model_inventory import load_inventory

ROOT = Path(__file__).resolve().parent
SHARED_INVENTORY = ROOT.parent / "modal-glm53flash-preflight" / "inventory-summary.json"
JSON_FILES = (
    "budget-plan.json",
    "experiment-manifest.json",
    "model-inventory-reference.json",
    "provider-snapshot.json",
)
ALLOWLIST = {
    "README.md",
    "SHA256SUMS",
    "budget-plan.json",
    "evidence_bundle.py",
    "experiment-manifest.json",
    "model-inventory-reference.json",
    "provider-snapshot.json",
    "report.html",
}
HASHED_FILES = tuple(sorted(ALLOWLIST - {"SHA256SUMS"}))

_PRIVATE_PATTERNS = (
    (re.compile("/" + "Users/"), "private home path"),
    (re.compile("/" + "home/"), "private home path"),
    (re.compile(r"\b[A-Za-z]:\\Users\\"), "private home path"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private SSH key"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "API credential"),
    (re.compile(r"\b(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b"), "IP address"),
    (re.compile(r"https?://[^/\s:@]+:[^@\s/]+@"), "URL credential"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "email address"),
)


class EvidenceError(ValueError):
    """Raised when committed public evidence is inconsistent or private."""


def _canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> dict[str, Any]:
    snapshot_data = _load("provider-snapshot.json")
    from llmtracefx.deploy.cloudrift import CloudRiftSnapshot

    snapshot = CloudRiftSnapshot.from_dict(snapshot_data)
    inventory = load_inventory(SHARED_INVENTORY)
    return build_cloudrift_plan(snapshot, inventory).to_dict()


def _render_report() -> str:
    manifest = _load("experiment-manifest.json")
    plan = _load("budget-plan.json")
    inventory = _load("model-inventory-reference.json")
    rows = (
        ("Decision", "refused before spend"),
        ("Authorized hard cap", "$80.00"),
        ("Conditional three-hour instance cost", "$60.00"),
        ("Required reserve", "$20.00"),
        ("Actual attributable spend", "$0.00 inferred"),
        ("CloudRift instances created", "0"),
        ("Available console shape", "8 x V100 SXM2, 16 GB each"),
        ("Available aggregate GPU memory", "128,000,000,000 bytes"),
        ("Model/GPU-memory shortfall", "200,366,172,318 bytes"),
        ("Available host memory", "52 GB"),
        ("Available disk", "400 GB"),
        ("Model revision", inventory["revision"]),
        ("Published inventory", f"{inventory['file_count']} files"),
        ("Published bytes", f"{inventory['total_bytes']:,}"),
        ("Published SHA-256 values", str(inventory["files_with_published_sha256"])),
        ("Readiness", "not run"),
        ("Smoke request", "not run"),
        ("Teardown", manifest["execution"]["teardown_status"]),
    )
    table = "\n".join(
        f"      <tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    blockers = "\n".join(
        f"      <li>{html.escape(blocker)}</li>" for blocker in plan["blockers"]
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>CloudRift GLM-5.3-Flash preflight refusal</title>
    <style>
      body {{ font: 16px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 920px; padding: 0 1rem; color: #17202a; }}
      h1 {{ line-height: 1.2; }}
      .refused {{ border-left: .4rem solid #b42318; background: #fef3f2; padding: 1rem; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border-bottom: 1px solid #d0d5dd; padding: .6rem; text-align: left; vertical-align: top; }}
      th {{ width: 42%; }}
      .cap {{ border: 1px solid #667085; height: 2rem; position: relative; }}
      .planned {{ background: #f79009; height: 100%; width: 75%; }}
      .cap span {{ position: absolute; inset: .25rem .5rem; font-weight: 700; }}
      code {{ overflow-wrap: anywhere; }}
    </style>
  </head>
  <body>
    <h1>CloudRift GLM-5.3-Flash preflight refusal</h1>
    <p class="refused"><strong>No paid lifecycle ran.</strong> Inventory,
      lifecycle, access, and recipe gates stopped the experiment before a
      CloudRift instance was created.</p>
    <h2>Conditional budget model</h2>
    <div class="cap" aria-label="$60 conditional plan within an $80 hard cap">
      <div class="planned"></div><span>$60 conditional plan / $80 hard cap</span>
    </div>
    <p>The $60 value assumes the advertised $2.50 rate is available on demand.
      Primary sources do not establish that assumption, so it is not spending
      authority.</p>
    <table>
{table}
    </table>
    <h2>Stop gates</h2>
    <ul>
{blockers}
    </ul>
    <h2>Supported conclusion</h2>
    <p>The immutable public model inventory remains available and unchanged.
      The only observed console shape was 8x V100 with 128 GB aggregate GPU
      memory, 200,366,172,318 bytes below the model inventory. This preflight
      does not establish staging, topology, model load, readiness, generation,
      performance, or billing.</p>
  </body>
</html>
"""


def render() -> None:
    (ROOT / "budget-plan.json").write_text(_canonical_json(_plan()), encoding="utf-8")
    (ROOT / "report.html").write_text(_render_report(), encoding="utf-8")
    hashes = [f"{_digest(ROOT / name)}  {name}" for name in HASHED_FILES]
    (ROOT / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")


def _scan_privacy(name: str, text: str) -> None:
    for pattern, description in _PRIVATE_PATTERNS:
        if pattern.search(text):
            raise EvidenceError(f"{name} contains {description}")


def verify() -> None:
    actual = {
        path.name for path in ROOT.iterdir() if not path.name.startswith("__pycache__")
    }
    if actual != ALLOWLIST:
        raise EvidenceError(f"unexpected bundle entries: {sorted(actual ^ ALLOWLIST)}")
    for name in JSON_FILES:
        data = _load(name)
        if (ROOT / name).read_text(encoding="utf-8") != _canonical_json(data):
            raise EvidenceError(f"{name} is not canonical deterministic JSON")

    manifest = _load("experiment-manifest.json")
    plan = _load("budget-plan.json")
    reference = _load("model-inventory-reference.json")
    if plan != _plan():
        raise EvidenceError("budget-plan.json differs from the offline planner")
    if plan["approved"] or manifest["decision"]["paid_execution_allowed"]:
        raise EvidenceError("refusal evidence must not approve paid execution")
    if plan["executable_steps"]:
        raise EvidenceError("the capability refusal exposes an executable step")
    authorization = plan["authorization"]
    if authorization["hard_cap_usd"] != "80.000000":
        raise EvidenceError("hard cap changed")
    if authorization["conditional_h200_cost_usd"] != "60.000000":
        raise EvidenceError("conditional planned cost changed")
    if authorization["minimum_reserve_usd"] != "20.000000":
        raise EvidenceError("required reserve changed")
    available = plan["available_configuration"]
    arithmetic = plan["capability_arithmetic"]
    if available["gpu_type"] != "V100 SXM2" or available["gpu_count"] != 8:
        raise EvidenceError("available CloudRift shape changed")
    if arithmetic["gpu_memory_shortfall_bytes"] != 200_366_172_318:
        raise EvidenceError("model/GPU-memory shortfall changed")
    if arithmetic["offload_or_substitution_allowed"]:
        raise EvidenceError("offload or substitution must remain refused")
    if manifest["provider_policy"]["cloudrift_paid_commands_executed"] != 0:
        raise EvidenceError("zero-spend evidence records a paid command")
    if manifest["provider_policy"]["instances_created"] != 0:
        raise EvidenceError("zero-spend evidence records an instance")
    if manifest["execution"]["smoke_requests_attempted"] != 0:
        raise EvidenceError("zero-spend evidence records a smoke request")

    shared_digest = _digest(SHARED_INVENTORY)
    if reference["source_bundle_sha256"] != shared_digest:
        raise EvidenceError("shared model inventory hash changed")
    inventory = load_inventory(SHARED_INVENTORY)
    inventory.assert_glm_53_flash()
    for field, value in inventory.summary().items():
        if field in reference and reference[field] != value:
            raise EvidenceError(f"inventory reference {field} changed")
    if reference["weights_downloaded_locally"]:
        raise EvidenceError("the inventory refresh must not download model weights")

    if (ROOT / "report.html").read_text(encoding="utf-8") != _render_report():
        raise EvidenceError("report.html is not the deterministic rendering")
    expected_hashes = {name: _digest(ROOT / name) for name in HASHED_FILES}
    recorded: dict[str, str] = {}
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    if recorded != expected_hashes:
        raise EvidenceError("SHA256SUMS does not match the public bundle")
    for name in sorted(ALLOWLIST):
        _scan_privacy(name, (ROOT / name).read_text(encoding="utf-8"))
    print("public CloudRift preflight evidence verified")


def main(argv: list[str]) -> int:
    command = argv[0] if argv else "verify"
    if command == "render":
        render()
        return 0
    if command == "verify":
        verify()
        return 0
    raise SystemExit("usage: evidence_bundle.py [render|verify]")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
