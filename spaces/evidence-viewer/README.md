---
title: LLMTraceFX Evidence Viewer
emoji: 🔎
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Inspect LLM inference evidence with provenance; export OTLP
tags:
  - llm
  - inference
  - benchmark
  - observability
  - opentelemetry
  - perfetto
---

# LLMTraceFX evidence viewer

Upload an [LLMTraceFX](https://github.com/Siddhant-K-code/LLMTraceFX)
`record.json` to see every measurement with its provenance, the run as a span
tree, and download it as a Perfetto trace or OTLP/JSON spans that follow the
OpenTelemetry GenAI semantic conventions. Optionally add the matching
`api_evidence.json` (streaming phases from `collect-api`) and
`server_metrics_evidence.json` (server-side queue and TTFT from Prometheus).

Files are processed in a temporary directory and not stored. Local model paths
are reduced to basenames in the exports.

## Deploying

This directory is the Space. From the repository root:

```bash
huggingface-cli login
huggingface-cli repo create llmtracefx-evidence-viewer --type space --space_sdk gradio
huggingface-cli upload <user>/llmtracefx-evidence-viewer spaces/evidence-viewer . --repo-type space
```

Run it locally with `pip install -r requirements.txt && python app.py`.
