# Releasing

Publishing a GitHub release for a tag `vX.Y.Z` runs two workflows:

- `publish.yml` builds the sdist and wheel from the tag, checks them (version
  matches the tag, `twine check --strict`, package data and the bootstrap
  script are in the wheel, the wheel installs and imports), and uploads them to
  [PyPI](https://pypi.org/project/llmtracefx/) with trusted publishing. It then
  moves the floating major tag (`v1` for 1.x) to the release commit, so
  `uses: Siddhant-K-code/LLMTraceFX@v1` gets the latest 1.x. The major tag only
  moves for the highest `vMAJOR.Y.Z` tag and never for a prerelease.
- `deploy-space.yml` waits for the version to appear on PyPI, then uploads
  `spaces/evidence-viewer/` to the Hugging Face Space with `requirements.txt`
  rewritten to `llmtracefx==X.Y.Z`.

Both workflows also run from the Actions tab (`workflow_dispatch`) with an
existing tag, for example to publish a release made before they existed or to
retry a failed run. PyPI never accepts the same file twice, so republishing a
version that is already uploaded fails without changing anything.

The tag must match `version` in `pyproject.toml`, or the publish fails before
anything is uploaded.

## One-time setup

### PyPI

1. On PyPI, open
   [Publishing](https://pypi.org/manage/account/publishing/) and add a pending
   trusted publisher with these values:

   | Field | Value |
   | --- | --- |
   | PyPI project name | `llmtracefx` |
   | Owner | `Siddhant-K-code` |
   | Repository name | `LLMTraceFX` |
   | Workflow name | `publish.yml` |
   | Environment name | `pypi` |

   The first successful upload creates the project and turns the pending
   publisher into a normal one.
2. In the repository settings, create an environment named `pypi`. Adding
   required reviewers there makes every upload wait for approval. If you limit
   its deployment refs, allow tags matching `v*` and `main`, since release runs
   use the tag and manual runs use the branch they start from.

No PyPI token is stored. Only the `publish` job can request an OIDC token, and
PyPI accepts it only from `publish.yml` in the `pypi` environment.

### Hugging Face Space

1. Create a Hugging Face access token with write access (a classic write token,
   or a fine-grained token that can create and write Spaces in the target
   namespace).
2. Add it as the repository secret `HF_TOKEN`.
3. Add the repository variable `HF_SPACE` with the Space id, for example
   `Siddhant-K-code/llmtracefx-evidence-viewer`.

The workflow creates the Space (Gradio SDK) if it does not exist. If either
`HF_TOKEN` or `HF_SPACE` is missing, the deploy is skipped with a notice and
the run still succeeds.

## Cutting a release

1. Bump the version in three places and merge to `main`:
   - `version` in `pyproject.toml`.
   - The `version` line of the `name = "llmtracefx"` package entry in
     `uv.lock`. Edit it by hand. Do not run `uv lock`, which rewrites
     `revision = 5`. If this line is stale, `uv sync --locked` in
     `publish.yml` fails and nothing is published.
   - The five `llmtracefx-X.Y.Z-py3-none-any.whl` filenames in
     `docs/VLLM_KV_TRUTH_RUNBOOK.md`.
2. Create a GitHub release with a new tag `vX.Y.Z` on that commit and publish it.
3. Check the "Publish to PyPI" and "Deploy Space" runs.

To publish an existing release, run "Publish to PyPI" and then "Deploy Space"
from the Actions tab with its tag.

`spaces/evidence-viewer/requirements.txt` in the repository installs from a
release tag on GitHub so the Space runs locally before a PyPI upload. The
deploy replaces that line, so it does not need bumping for each release.
