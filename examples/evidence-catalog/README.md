# LLMTraceFX offline evidence catalog

This directory is generated from the closed registry in
`llmtracefx/evidence/registry.py`. Catalog metadata is not trusted: verification
revalidates the strict schema, catalog hash, exact bundle allowlists, every bundle's
closed verifier adapter, privacy gates, content identities, and lineage graph.

Reproduce from a source checkout without network access, credentials, model loads,
cloud authentication, or paid execution:

```bash
uv run llmtracefx-evidence index
uv run llmtracefx-evidence verify
uv run llmtracefx-evidence graph
make evidence-catalog
```

From an installed wheel and unrelated working directory, pass the committed catalog
explicitly; the repository root is inferred from that path:

```bash
llmtracefx-evidence verify --catalog /path/to/repo/examples/evidence-catalog/catalog.json
```

`catalog.json` is canonical machine-readable metadata. `graph.json` and `graph.dot`
carry the same typed lineage. `graph.svg` and `index.html` are self-contained static
views. `claim-matrix.json` preserves supported, unsupported, and not-applicable
claim states with provenance. `SHA256SUMS` covers every other generated file and intentionally excludes itself
to avoid a circular checksum.

Unknown example directories are never inferred into the catalog. Candidate evidence
directories not present in the closed registry appear in `unregistered_candidates`
and make verification fail until explicitly reviewed and registered.
