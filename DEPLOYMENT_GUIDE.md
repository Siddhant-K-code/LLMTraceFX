# Deployment documentation

The deployment guide that previously lived at this path described an older
public Modal analyzer endpoint. That endpoint, its dashboard, and its optional
explanation workflow are not the current LLMTraceFX product path. The endpoint
is retired, so its URL and request examples have been removed.

Use the current documentation instead:

- [README quickstart](README.md#quickstart) for offline, no-credential setup.
- [Modal GLM-5.3-Flash runbook](SELF_HOST_GLM_RUNBOOK.md) for the optional
  budget-guarded planning and deployment lifecycle.
- `uv run llmtracefx-deploy --help` for the installed planning CLI.

`llmtracefx-deploy recipe`, `budget`, and `plan` do not authenticate, access the
network, download a model, create a Modal resource, or allocate an accelerator.
Commands after the paid execution boundary in the runbook can incur charges.
