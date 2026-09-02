"""Render and verify the public Modal GLM preflight refusal bundle."""

from __future__ import annotations

import hashlib
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
JSON_FILES = (
    "experiment-manifest.json",
    "inventory-summary.json",
    "pricing.json",
    "budget-plan.json",
)
ALLOWLIST = {
    "README.md",
    "SHA256SUMS",
    "budget-plan.json",
    "evidence_bundle.py",
    "experiment-manifest.json",
    "inventory-summary.json",
    "pricing.json",
    "report.html",
}
HASHED_FILES = tuple(sorted(ALLOWLIST - {"SHA256SUMS"}))


class EvidenceError(ValueError):
    """Raised when committed evidence is inconsistent or unsafe to publish."""


_PRIVATE_PATTERNS = (
    (re.compile("/" + "Users/"), "private home path"),
    (re.compile(r"\b(?:wk|ws)-[A-Za-z0-9_-]{6,}\b"), "Modal credential"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"), "API credential"),
    (re.compile(r"https://[^\s\"']+\.modal\.run"), "private Modal endpoint"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "email address"),
)


def _canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True) + "\n"


def _load(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _scan_privacy(name: str, text: str) -> None:
    for pattern, description in _PRIVATE_PATTERNS:
        if pattern.search(text):
            raise EvidenceError(f"{name} contains {description}")


def _render_report() -> str:
    manifest = _load("experiment-manifest.json")
    inventory = _load("inventory-summary.json")
    pricing = _load("pricing.json")
    plan = _load("budget-plan.json")
    rows = [
        ("Decision", manifest["decision"]["status"]),
        ("Authorized hard cap", "$10.00"),
        (
            "Modeled complete envelope",
            f"${plan['cost_envelope']['worst_case_usd']:.2f}",
        ),
        ("Observed Modal spend", "$0.00"),
        ("Modal CLI authenticated", "no"),
        ("Paid commands executed", "0"),
        ("Model revision", manifest["model"]["revision"]),
        ("Published inventory", f"{inventory['file_count']} files"),
        ("Published bytes", f"{inventory['total_bytes']:,}"),
        ("SHA-256 metadata", f"{inventory['files_with_published_sha256']} files"),
        ("Readiness", "not run"),
        ("Smoke request", "not run"),
        ("Teardown", "not required; no resources created"),
    ]
    table = "\n".join(
        f"      <tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"
        for label, value in rows
    )
    sources = "\n".join(
        f'      <li><a href="{html.escape(url)}">{html.escape(url)}</a></li>'
        for url in (
            inventory["source"],
            pricing["pricing_source"],
            pricing["volume_policy_source"],
            "https://huggingface.co/zai-org/GLM-5.3-Flash",
            "https://recipes.vllm.ai/zai-org/GLM-5.3-Flash",
        )
    )
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Modal GLM-5.3-Flash preflight refusal</title>
    <style>
      body {{ font: 16px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 900px; padding: 0 1rem; color: #17202a; }}
      h1 {{ line-height: 1.2; }}
      .refused {{ border-left: .4rem solid #b42318; background: #fef3f2; padding: 1rem; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border-bottom: 1px solid #d0d5dd; padding: .6rem; text-align: left; vertical-align: top; }}
      th {{ width: 34%; }}
      code {{ overflow-wrap: anywhere; }}
    </style>
  </head>
  <body>
    <h1>Modal GLM-5.3-Flash preflight refusal</h1>
    <p class="refused"><strong>No paid lifecycle ran.</strong> Authentication,
      budget, and framework-provenance gates stopped the experiment before any
      Modal resource was created.</p>
    <table>
{table}
    </table>
    <h2>Directly supported conclusion</h2>
    <p>The exact public checkpoint and its published inventory can be pinned,
      and four requested H200s provide arithmetic nameplate capacity above the
      published weight size. This preflight does not establish that the model
      stages, verifies, loads, becomes ready, or serves an authenticated output.</p>
    <h2>Sources</h2>
    <ul>
{sources}
    </ul>
  </body>
</html>
"""


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render() -> None:
    (ROOT / "report.html").write_text(_render_report(), encoding="utf-8")
    lines = [f"{_digest(ROOT / name)}  {name}" for name in HASHED_FILES]
    (ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify() -> None:
    actual = {
        path.name for path in ROOT.iterdir() if not path.name.startswith("__pycache__")
    }
    if actual != ALLOWLIST:
        raise EvidenceError(f"unexpected bundle entries: {sorted(actual ^ ALLOWLIST)}")
    for name in JSON_FILES:
        path = ROOT / name
        data = json.loads(path.read_text(encoding="utf-8"))
        normalized = path.read_text(encoding="utf-8").rstrip() + "\n"
        if normalized != _canonical_json(data):
            raise EvidenceError(f"{name} is not canonical deterministic JSON")
    manifest = _load("experiment-manifest.json")
    plan = _load("budget-plan.json")
    if manifest["decision"]["paid_execution_allowed"]:
        raise EvidenceError("manifest must preserve the paid refusal")
    if plan["approved"]:
        raise EvidenceError("budget plan must preserve the planner refusal")
    paid = {step["name"] for step in plan["steps"] if step["spends_money"]}
    if paid.intersection(plan["executable_steps"]):
        raise EvidenceError("a paid step escaped the refused plan")
    if manifest["authorization"]["actual_or_observed_credit_use_usd"] != "0.000000":
        raise EvidenceError("preflight evidence must not claim provider spend")
    expected_report = _render_report()
    if (ROOT / "report.html").read_text(encoding="utf-8") != expected_report:
        raise EvidenceError("report.html is not the deterministic JSON rendering")
    expected_hashes = {name: _digest(ROOT / name) for name in HASHED_FILES}
    recorded: dict[str, str] = {}
    for line in (ROOT / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        recorded[name] = digest
    if recorded != expected_hashes:
        raise EvidenceError("SHA256SUMS does not match the committed bundle")
    for name in sorted(ALLOWLIST):
        _scan_privacy(name, (ROOT / name).read_text(encoding="utf-8"))
    print("public Modal preflight evidence verified")


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
