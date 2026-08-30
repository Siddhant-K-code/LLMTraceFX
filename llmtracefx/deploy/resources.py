"""The container shapes the harness actually asks Modal for.

These live in one module because two different things need to agree on
them and the cost of disagreement is a wrong number rather than an
error: the Modal entrypoint requests these resources, and the budget
prices them. If the entrypoint asked for sixteen cores while the
envelope priced eight, the plan would approve a deployment that costs
more than it said, and nothing would fail until the invoice.

Every value here is therefore imported by both sides. None of them is
configurable, because a knob that changes what is billed without
changing what is priced is the same bug with more steps.
"""

from __future__ import annotations

# The staging container. CPU and network only; sized for parallel
# downloads of a few hundred GiB rather than for compute.
STAGING_CPU_CORES = 8.0
STAGING_MEMORY_GIB = 32.0
STAGING_TIMEOUT_SECONDS = 6 * 60 * 60

# The verification container. Reads the staged tree back and hashes it,
# so it wants throughput rather than cores.
VERIFY_CPU_CORES = 4.0
VERIFY_MEMORY_GIB = 16.0
VERIFY_TIMEOUT_SECONDS = 6 * 60 * 60

# The serving container, alongside its accelerators. Requested
# explicitly rather than left to the platform default, because a default
# is a number nobody wrote down and therefore a number the envelope
# cannot price.
SERVING_CPU_CORES = 16.0
SERVING_MEMORY_GIB = 64.0

# Modal bills volume storage for a period after deletion:
# "When you delete data, you may still be billed for that storage for up
# to four days" (https://modal.com/docs/guide/volumes, read 2026-08-30).
# The worst case adds this to whatever retention the operator declares,
# because it is charged whether or not they remember it.
POST_DELETE_BILLING_DAYS = 4.0
