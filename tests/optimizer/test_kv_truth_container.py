from __future__ import annotations

from pathlib import Path

CONTAINERFILE = (
    Path(__file__).resolve().parents[2] / "containers/vllm-kv-truth/Containerfile"
)
BASE_IMAGE = (
    "vllm/vllm-openai:v0.28.0@"
    "sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635"
)


def test_containerfile_is_bound_to_the_exact_offline_runtime() -> None:
    text = CONTAINERFILE.read_text(encoding="utf-8")
    assert text.startswith(f"FROM {BASE_IMAGE}\n")
    assert "--no-index" in text
    assert "--no-deps" in text
    assert "HF_HUB_OFFLINE=1" in text
    assert "TRANSFORMERS_OFFLINE=1" in text
    assert "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0" in text
    assert "PYTHONHASHSEED=0" in text
    assert "COPY . /opt/llmtracefx/source" in text
    assert "--no-build-isolation" in text
    assert 'org.llmtracefx.protocol="qwen3-8b-vllm-kv-truth-v1"' in text
    assert 'org.llmtracefx.downloader.package="huggingface-hub"' in text
    assert 'org.llmtracefx.downloader.version="1.13.0"' in text
    assert (
        'org.llmtracefx.downloader.interface="huggingface_hub.snapshot_download"'
        in text
    )


def test_containerfile_has_no_download_or_public_service_step() -> None:
    text = CONTAINERFILE.read_text(encoding="utf-8").lower()
    for forbidden in (
        "curl ",
        "wget ",
        "apt-get",
        "git clone",
        "huggingface-cli",
        "expose ",
    ):
        assert forbidden not in text
