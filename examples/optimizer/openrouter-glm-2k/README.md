# OpenRouter GLM 2K hosted comparison

This privacy-safe bundle records eight paid OpenRouter requests made on
2026-09-02: two measured repetitions of the pinned 2K structured-JSON and
prose-reasoning workloads for `z-ai/glm-5.3-flash` and `z-ai/glm-5.3`.
No warmup or retry requests were made. Routing was fixed to OpenRouter's
first-party `z-ai/fp8` endpoint, fallbacks were disabled, and generation
metadata verified the resolved Z.AI builds.

Both systems passed all four evaluated cases. Under the single objective of
maximizing correct cases per minute after a 100% pass-rate gate, GLM-5.3 was
faster in both exact workload strata:

| Workload | GLM-5.3 | GLM-5.3-Flash |
| --- | ---: | ---: |
| Structured JSON | 18.296 correct/min; 3,279 ms mean | 9.836 correct/min; 6,100 ms mean |
| Prose reasoning | 18.096 correct/min; 3,316 ms mean | 6.382 correct/min; 9,402 ms mean |

Flash was substantially cheaper. Manifest-estimated totals from
provider-reported usage were $0.00015343 vs. $0.00282232 for structured JSON
and $0.00024615 vs. $0.00293072 for prose (Flash vs. GLM-5.3). This is not a
universal winner: GLM-5.3 won the declared throughput objective while Flash
produced more correct cases per dollar.

The application ledger authorized at most $5.00, conservatively reserved at
most $0.0745326 for the full matrix, and recorded $0.00615262 from the eight
final provider usage blocks. The immediate post-run account query showed only
$0.00039958, equal to the Flash calls; account usage can lag and can include
unrelated activity, so that delta is kept separate rather than treated as the
experiment total.

The merged local Qwen3-8B 2K control at commit
`a6077adaf7135e2a2e360aeae4a73b6b411b3493` is contextual only. It was
excluded from direct ranking because it is a different model/system identity,
used local MLX timing, disabled thinking, used a seed the pinned hosted
endpoint did not advertise, and was exploratory without a clean-boot
assertion.

The bundle contains hashes, plans, sanitized records, usage, timing, budget
state, comparison JSON/HTML, and limitations. It contains no prompt,
response, reasoning text, credential, account identifier, raw header,
provider request identifier, or private path.

## Verify

```bash
python examples/optimizer/openrouter-glm-2k/evidence_bundle.py verify
```

## Separate self-hosted question

A bounded Modal run would answer a materially different systems question:
whether the exact GLM-5.3-Flash checkpoint loads and fits under selected
hardware/runtime controls, and how tensor parallelism, load time, memory, and
server-side throughput behave without OpenRouter's gateway and provider
queueing. It should not be used to "confirm" these hosted latency numbers.

If separately authorized, the smallest useful plan is the existing offline
recipe/budget/plan flow followed by one cold load and the same two safe 2K
workloads with one warmup and two measured repetitions, a single pinned GPU
configuration, no autoscaling beyond one container, an explicit short idle
timeout, a hard session spending cap, and stop-on-first-failure. No Modal
authentication, deployment, weight download, or spend occurred in this task.
