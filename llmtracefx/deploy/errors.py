"""Errors raised while planning a self-hosted deployment.

A single error type keeps the failure mode uniform: every unsafe,
unaffordable, unpinned or under-specified plan is refused the same way,
before anything is launched, so a caller never has to distinguish
"refused because it costs too much" from "refused because the model
revision was not pinned" in order to know that nothing was spent.
"""

from __future__ import annotations


class DeploymentPlanError(ValueError):
    """A deployment plan is unsafe, unaffordable, unpinned or incomplete.

    Raised only during planning, which is a pure, offline computation, so
    receiving this exception is a positive guarantee that no paid resource
    was allocated and no network call was made.
    """
