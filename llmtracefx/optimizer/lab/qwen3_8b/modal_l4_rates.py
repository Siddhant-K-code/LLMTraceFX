"""Rate refresh capture and account-headroom validation for the Modal delta.

Nothing here reaches the network unless a caller passes ``fetch=True`` on
an execution path. The offline and test paths take an injected fetcher or
an already-captured document, so the whole surface is exercisable with no
socket, no credential, and no provider account.

The design point is that a pricing page is prose. Parsing prose into the
number a budget depends on is exactly where a silent, expensive mistake
lives, so this module never derives rates from HTML. It captures and
hashes the raw official document as provenance, and requires the rates
themselves to arrive as an exact structured receipt that is then checked
against the committed rates. A higher official rate or a new mandatory
charge is a refusal, not a re-plan.
"""

from __future__ import annotations

import hashlib
import json
import ssl
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from ..._artifact_io import (
    MAX_METADATA_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from .modal_l4_crossover import (
    OFFICIAL_RATE_DOMAINS,
    OFFICIAL_RATE_URL,
    TOTAL_PLANNED_USD,
    ModalL4ContractError,
    verify_official_rate_receipt,
)
from .vllm_compile import canonical_decimal

OFFICIAL_VOLUME_RATE_URL = "https://modal.com/docs/guide/volumes"
OFFICIAL_SOURCE_URLS = (OFFICIAL_RATE_URL, OFFICIAL_VOLUME_RATE_URL)
MAX_DOCUMENT_BYTES = 4 * 1024 * 1024
FETCH_TIMEOUT_SECONDS = 20
USER_AGENT = "llmtracefx-modal-l4-rate-refresh"
HEADROOM_SIGNATURE_NAMESPACE = "llmtracefx-modal-l4-headroom-v1"


class RateRefreshError(ModalL4ContractError):
    """Raised when official rate provenance cannot be established."""


class DocumentFetcher(Protocol):
    def __call__(self, url: str) -> bytes: ...


def _sha256_uri(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _require_official(url: str) -> str:
    if not isinstance(url, str) or not url.startswith("https://"):
        raise RateRefreshError("rate source must be an https URL")
    host = url.split("/")[2].split("@")[-1].split(":")[0].lower()
    if not any(
        host == domain or host.endswith(f".{domain}")
        for domain in OFFICIAL_RATE_DOMAINS
    ):
        raise RateRefreshError(f"rate source {host!r} is not an official domain")
    return url


def https_document_fetcher(url: str) -> bytes:
    """Fetch one official document over TLS, sending no credential.

    No cookie, no authorization header, no Modal profile, and no
    redirect to a non-official host. The response is bounded so a large
    or hostile document cannot be read into memory unchecked.
    """

    _require_official(url)
    request = urllib.request.Request(  # noqa: S310 - scheme and host are checked
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,text/plain"},
        method="GET",
    )
    context = ssl.create_default_context()
    try:
        with urllib.request.urlopen(  # noqa: S310 - scheme and host are checked
            request, timeout=FETCH_TIMEOUT_SECONDS, context=context
        ) as response:
            _require_official(response.geturl())
            data = bytes(response.read(MAX_DOCUMENT_BYTES + 1))
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RateRefreshError(
            f"official rate document could not be fetched: {type(exc).__name__}"
        ) from exc
    if len(data) > MAX_DOCUMENT_BYTES:
        raise RateRefreshError("official rate document exceeds its bound")
    if not data:
        raise RateRefreshError("official rate document is empty")
    return data


def capture_rate_documents(
    *,
    fetcher: DocumentFetcher,
    urls: Sequence[str] = OFFICIAL_SOURCE_URLS,
    observed_at: str,
) -> dict[str, Any]:
    """Capture and hash the official documents without interpreting them."""

    if not urls:
        raise RateRefreshError("at least one official rate document is required")
    documents = []
    for url in urls:
        data = fetcher(_require_official(url))
        if not isinstance(data, bytes) or not data:
            raise RateRefreshError("official rate document is empty")
        documents.append({"url": url, "bytes": len(data), "sha256": _sha256_uri(data)})
    return {
        "kind": "modal_rate_document_capture",
        "observed_at": observed_at,
        "documents": documents,
        "capture_sha256": _sha256_uri(
            json.dumps(
                documents, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ),
        "parsed_from_html": False,
        "parsing_limitation": (
            "official rates are never derived from page markup; the capture is "
            "provenance for an exact structured receipt"
        ),
    }


def verify_rate_refresh(
    receipt: Any,
    *,
    capture: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a structured rate receipt to a freshly captured official page."""

    if not isinstance(receipt, Mapping):
        raise RateRefreshError("rate receipt must be an object")
    if not isinstance(capture, Mapping) or not capture.get("documents"):
        raise RateRefreshError("rate capture is missing")
    captured = {
        str(item["url"]): str(item["sha256"])
        for item in capture["documents"]
        if isinstance(item, Mapping)
    }
    source_url = receipt.get("source_url")
    if source_url not in captured:
        raise RateRefreshError("rate receipt is not bound to a captured document")
    if receipt.get("document_sha256") != captured[source_url]:
        raise RateRefreshError(
            "rate receipt hash differs from the freshly captured document"
        )
    verified = verify_official_rate_receipt(receipt)
    return {
        **verified,
        "capture_sha256": capture["capture_sha256"],
        "captured_documents": sorted(captured),
        "parsed_from_html": False,
    }


def refresh_official_rates(
    *,
    structured_receipt: Mapping[str, Any],
    observed_at: str,
    fetcher: DocumentFetcher | None = None,
    urls: Sequence[str] = OFFICIAL_SOURCE_URLS,
) -> dict[str, Any]:
    """Capture, hash, and adjudicate the official rates before any spend."""

    capture = capture_rate_documents(
        fetcher=fetcher or https_document_fetcher,
        urls=urls,
        observed_at=observed_at,
    )
    return {
        "capture": capture,
        "verification": verify_rate_refresh(structured_receipt, capture=capture),
    }


def read_structured_receipt(path: Any) -> dict[str, Any]:
    """Read an operator-supplied structured rate receipt safely."""

    try:
        payload = json.loads(
            read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES),
            parse_constant=reject_non_finite_json_constant,
        )
    except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
        # The raised exception can carry the operator's filesystem path or a
        # snippet of the unsafe document. Only the failure category is
        # surfaced so nothing path- or content-derived is emitted or persisted.
        raise RateRefreshError(
            "rate receipt could not be read safely: " f"{type(exc).__name__}"
        ) from exc
    if not isinstance(payload, dict):
        raise RateRefreshError("rate receipt must be an object")
    return payload


def account_headroom(
    *,
    control_plane_probe: Any = None,
    signed_receipt: Mapping[str, Any] | None = None,
    signature_verifier: Any = None,
) -> dict[str, Any]:
    """Establish pre-run spend headroom, or refuse to infer it.

    The provider SDK exposes post-hoc workspace billing, not a pre-run
    spend authority. When no probe is available the only acceptable
    substitute is a separately signed operator receipt: an absent limit is
    never read as an unlimited one.
    """

    if control_plane_probe is not None:
        observed = control_plane_probe()
        if not isinstance(observed, Mapping) or "headroom_usd" not in observed:
            raise RateRefreshError(
                "control-plane headroom probe returned no sanitized headroom"
            )
        headroom = observed["headroom_usd"]
        if not isinstance(headroom, str):
            raise RateRefreshError("control-plane headroom must be a decimal string")
        return _adjudicate_headroom(
            headroom, provenance="modal_control_plane_sanitized_probe"
        )
    if signed_receipt is None:
        raise RateRefreshError(
            "no headroom probe and no signed operator receipt; refusing to "
            "infer account headroom"
        )
    if signature_verifier is None:
        raise RateRefreshError("a signed headroom receipt requires a verifier")
    signature_verifier(dict(signed_receipt))
    for field in ("account_identifier", "account_id", "workspace", "email"):
        if field in signed_receipt:
            raise RateRefreshError(
                "headroom receipt must be sanitized of account identity"
            )
    headroom = signed_receipt.get("headroom_usd")
    if not isinstance(headroom, str):
        raise RateRefreshError("headroom receipt must carry a decimal string")
    return _adjudicate_headroom(
        headroom,
        provenance="signed_operator_receipt",
        namespace=(HEADROOM_SIGNATURE_NAMESPACE),
    )


def _adjudicate_headroom(
    headroom: str,
    *,
    provenance: str,
    namespace: str | None = None,
) -> dict[str, Any]:
    from decimal import Decimal, InvalidOperation

    try:
        value = Decimal(headroom)
    except InvalidOperation as exc:
        raise RateRefreshError("headroom must be a canonical decimal string") from exc
    if not value.is_finite() or value < 0 or canonical_decimal(value) != headroom:
        raise RateRefreshError("headroom must be a canonical, finite, non-negative sum")
    if value < TOTAL_PLANNED_USD:
        raise RateRefreshError(
            "account headroom is below the planned total; refusing to start"
        )
    return {
        "supported": True,
        "headroom_usd": canonical_decimal(value),
        "provenance": provenance,
        "signature_namespace": namespace,
        "is_provider_spend_proof": False,
        "null_reason": None,
    }
