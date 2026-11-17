"""Replay published LLM arrival traces against an OpenAI-compatible endpoint.

Supported inputs are Mooncake JSONL (timestamp, input_length, output_length,
hash_ids), BurstGPT CSV and AIPerf ``single_turn`` JSONL with absolute
timestamps. Each file is parsed strictly: a malformed row refuses the whole
trace rather than being skipped or repaired. The schedule layer turns the
accepted rows into timestamped requests with synthetic prompts that keep the
trace's prefix-sharing structure, and records the trace sha256, the
selection and any time scaling it applied.
"""

from .formats import TRACE_FORMATS, ParsedTrace, TraceEntry, TraceFormatError, parse_trace
from .schedule import ReplaySchedule, ScheduleError, Selection, build_schedule

__all__ = [
    "TRACE_FORMATS",
    "ParsedTrace",
    "ReplaySchedule",
    "ScheduleError",
    "Selection",
    "TraceEntry",
    "TraceFormatError",
    "build_schedule",
    "parse_trace",
]
