"""LLMTraceFX evidence viewer: a Hugging Face Space.

Upload an LLMTraceFX ``record.json`` (and optionally the matching
``api_evidence.json`` and server metrics evidence) to see every measurement
with its provenance, and download the run as a Perfetto trace or OTLP/JSON
spans. Nothing is stored; files are processed in a temporary directory.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

try:
    # Preinstalled on ZeroGPU Spaces. Import before gradio.
    import spaces
except ImportError:
    spaces = None

import gradio as gr
import gradio.queueing
import gradio.utils
from viewer import ViewerError, render


def _lock() -> asyncio.Lock:
    return asyncio.Lock()


def _stop_event() -> asyncio.Event:
    return asyncio.Event()


# gradio's safe_get_lock and safe_get_stop_event create and set a new event
# loop on every call made outside a running loop and never close the old one.
# When the garbage collector later finalizes such a loop after its self-pipe
# socket, Python logs "ValueError: Invalid file descriptor: -1" from
# BaseEventLoop.__del__ (gradio-app/gradio#12699). On Python 3.10+ asyncio.Lock
# and asyncio.Event bind to the running loop on first use, so no loop is needed.
for _module, _name, _replacement in (
    (gradio.utils, "safe_get_lock", _lock),
    (gradio.utils, "safe_get_stop_event", _stop_event),
    (gradio.queueing, "safe_get_lock", _lock),
):
    if hasattr(_module, _name):
        setattr(_module, _name, _replacement)

EXAMPLE = Path(__file__).parent / "examples" / "synthetic-record.json"

if spaces is not None:
    # ZeroGPU refuses to start an app with no @spaces.GPU function. The viewer
    # needs no GPU, so this is never called and uses no GPU quota.
    @spaces.GPU
    def _zerogpu_placeholder() -> None:
        return None


def view(record, api_evidence, server_metrics):
    if record is None:
        raise gr.Error("Upload a record.json first.")
    try:
        result = render(record, api_evidence, server_metrics)
    except ViewerError as exc:
        raise gr.Error(str(exc)) from exc
    return (
        result.summary_markdown,
        result.measurements_markdown,
        result.spans_markdown,
        str(result.perfetto_path),
        str(result.otlp_path),
    )


with gr.Blocks(title="LLMTraceFX evidence viewer") as demo:
    gr.Markdown(
        "# LLMTraceFX evidence viewer\n"
        "Inspect an [LLMTraceFX](https://github.com/Siddhant-K-code/LLMTraceFX) "
        "run: every value is shown with its provenance (measured natively, "
        "wall clock, provider reported, or derived). Export the run to "
        "[Perfetto](https://ui.perfetto.dev) or any OpenTelemetry backend.\n\n"
        "Produce records with `llmtracefx-optimizer collect-api`, "
        "`collect-mlx`, or `parse-llama-cpp`. The bundled example is "
        "synthetic, not a benchmark result."
    )
    with gr.Row():
        record = gr.File(label="record.json", file_types=[".json"], type="filepath")
        api_evidence = gr.File(
            label="api_evidence.json (optional)", file_types=[".json"], type="filepath"
        )
        server_metrics = gr.File(
            label="server_metrics_evidence.json (optional)",
            file_types=[".json"],
            type="filepath",
        )
    button = gr.Button("View evidence", variant="primary")
    summary = gr.Markdown()
    with gr.Row():
        measurements = gr.Markdown()
        spans = gr.Markdown()
    with gr.Row():
        perfetto = gr.File(label="Perfetto / Chrome trace (open in ui.perfetto.dev)")
        otlp = gr.File(label="OTLP/JSON spans (GenAI semantic conventions)")
    outputs = [summary, measurements, spans, perfetto, otlp]
    button.click(view, [record, api_evidence, server_metrics], outputs)
    gr.Examples(
        examples=[[str(EXAMPLE), None, None]],
        inputs=[record, api_evidence, server_metrics],
        outputs=outputs,
        fn=view,
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch()
