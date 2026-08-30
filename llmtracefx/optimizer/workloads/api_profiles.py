"""Named, documented endpoint profiles for OpenAI-compatible providers.

A profile is *data*, not behavior. It records the three things that are
tedious and error-prone to retype for a given provider (the full
chat-completions URL, the conventional credential environment variable
name, and the model IDs that provider publishes) and nothing else. No
module in this package branches on which profile was selected: a profile
only supplies defaults that the caller may override, so pointing the API
workload runner at an unlisted provider stays a pure configuration
change with no code change and no second-class behavior.

Model IDs are recorded as ``documented_model_ids`` purely so that
``--list-profiles`` can show a caller what the provider publishes. They
are never validated against, because a provider adds models faster than
this file can be edited and rejecting an unlisted model would turn a
documentation aid into a gate.

The endpoints and model IDs below come from each provider's published
documentation:

* OpenRouter: https://openrouter.ai/docs/api-reference/overview
* Z.ai: https://docs.z.ai/api-reference/llm/chat-completion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class APIProfileError(ValueError):
    """Raised when an unknown provider profile name is requested."""


@dataclass(frozen=True)
class APIProfile:
    """Documented defaults for one OpenAI-compatible chat endpoint."""

    name: str
    """Profile selector, e.g. ``openrouter``."""

    provider_label: str
    """Short sanitized provider label recorded in evidence."""

    endpoint: str
    """Full chat-completions URL."""

    credential_env_var: str
    """Conventional environment variable holding the API key. Only the
    name is ever used or persisted; the value is read by the collector
    straight from the environment."""

    documented_model_ids: tuple[str, ...]
    """Model IDs the provider documents for this endpoint. Informational
    only: an unlisted model ID is accepted without complaint."""

    notes: str
    """One-line description shown by ``--list-profiles``."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider_label": self.provider_label,
            "endpoint": self.endpoint,
            "credential_env_var": self.credential_env_var,
            "documented_model_ids": list(self.documented_model_ids),
            "notes": self.notes,
        }


OPENROUTER_PROFILE = APIProfile(
    name="openrouter",
    provider_label="openrouter",
    endpoint="https://openrouter.ai/api/v1/chat/completions",
    credential_env_var="OPENROUTER_API_KEY",
    documented_model_ids=("z-ai/glm-5.3", "z-ai/glm-5.3-flash"),
    notes=(
        "OpenRouter's OpenAI-compatible gateway; GLM models are addressed "
        "with the z-ai/ vendor prefix"
    ),
)

ZAI_PROFILE = APIProfile(
    name="z.ai",
    provider_label="z.ai",
    endpoint="https://api.z.ai/api/paas/v4/chat/completions",
    credential_env_var="ZAI_API_KEY",
    documented_model_ids=("glm-5.3", "glm-5.3-flash"),
    notes="Z.ai's first-party GLM chat-completions endpoint",
)

API_PROFILES: tuple[APIProfile, ...] = (OPENROUTER_PROFILE, ZAI_PROFILE)

PROFILE_NAMES: tuple[str, ...] = tuple(profile.name for profile in API_PROFILES)


def profile_by_name(name: str) -> APIProfile:
    """Return the profile called ``name``.

    Raises ``APIProfileError`` listing the known names, which keeps the
    diagnostic useful without echoing the caller's value; an unknown
    profile name is a typo, not a credential, but the same restraint is
    applied here as everywhere else in this package.
    """
    for profile in API_PROFILES:
        if profile.name == name:
            return profile
    raise APIProfileError(
        "unknown provider profile; known profiles are "
        + ", ".join(PROFILE_NAMES)
        + ". A profile only supplies defaults, so an unlisted provider can "
        "still be measured by passing --endpoint, --provider and "
        "--api-key-env explicitly."
    )
