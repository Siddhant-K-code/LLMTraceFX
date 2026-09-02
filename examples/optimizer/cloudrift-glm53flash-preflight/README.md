# CloudRift GLM-5.3-Flash preflight

This bundle records a **zero-spend refusal** for one proposed 8x H200,
tensor-parallel-8 smoke validation. The authenticated CloudRift console
inventory available to the user offered only 8x V100 SXM2 with 16 GB per GPU,
52 GB host RAM, 400 GB disk, and a listed $0.25/GPU-hour. It does not report a
model run, and identifiers were not collected.

The conditional arithmetic is $2.50/GPU-hour x 8 GPUs x 3 hours = $60,
leaving $20 of the authorized $80 hard cap. That arithmetic is not executable
pricing: CloudRift's H200 table marks the product pre-order and associates
$2.50 with a one-month term, while no short on-demand H200 rate or account
inventory was verified.

Paid execution was also stopped because the public renter API exposes no
scheduled termination/TTL, monetary rounding and credit behavior are
incomplete, the documented `rift` access was not configured locally, and
CloudRift's only GLM recipe is untested and does not pin this checkpoint,
vLLM, TP=8, FP8, or a container digest.

The capability mismatch is independently decisive. Eight listed V100s provide
128,000,000,000 bytes of aggregate advertised GPU memory, which is
200,366,172,318 bytes below the exact published model inventory. Even adding
the 52 GB host RAM arithmetically reaches only 180,000,000,000 bytes. Offload,
another model, another quantization, or repurposing the $80 authorization was
not allowed.

The exact immutable inventory is reused from the merged Modal preflight rather
than copied: 72 files, 328,366,172,318 bytes, 62 safetensors shards, and 63
published SHA-256 values at revision
`03eb5366286afd40d2221b1d9c63a6dd1ba4832e`. Its upstream metadata was
re-fetched; no model bytes were downloaded locally.

## Verify offline

```bash
uv run python examples/optimizer/cloudrift-glm53flash-preflight/evidence_bundle.py verify
```

## Primary sources

- [CloudRift pricing and FAQ](https://www.cloudrift.ai/pricing)
- [CloudRift terms](https://www.cloudrift.ai/terms)
- [CloudRift public OpenAPI](https://api.cloudrift.ai/api-docs/openapi.json)
- [CloudRift CLI setup](https://docs.cloudrift.ai/setup/cli)
- [CloudRift instance discovery](https://docs.cloudrift.ai/cli-interface/instance-management)
- [CloudRift VM lifecycle](https://docs.cloudrift.ai/cli-interface/vm-management)
- [CloudRift persistent volumes](https://docs.cloudrift.ai/features/volumes)
- [CloudRift volume billing troubleshooting](https://docs.cloudrift.ai/troubleshooting/volumes-and-reservations)
- [CloudRift setup and billing troubleshooting](https://docs.cloudrift.ai/troubleshooting/setup-and-general)
- [Pinned CloudRift GLM recipe](https://github.com/cloudrift-ai/emmy/blob/ec0ca0213378dcba6d3ea2cd16ba66e1a34c2c9c/recipes/glm-5-3/recipe.yaml)
- [Pinned Hugging Face inventory](https://huggingface.co/api/models/zai-org/GLM-5.3-Flash/revision/03eb5366286afd40d2221b1d9c63a6dd1ba4832e?blobs=true)
