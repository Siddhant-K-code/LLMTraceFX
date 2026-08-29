"""Tests for the incremental SSE decoder used by the streaming API collector.

These tests exercise byte-level framing only. No network is involved and
no provider is contacted.
"""

from __future__ import annotations

import pytest

from llmtracefx.optimizer.collectors.sse import SSEDecodeError, SSEDecoder, SSEEvent


def drain(decoder: SSEDecoder, chunks: list[bytes]) -> list[SSEEvent]:
    events: list[SSEEvent] = []
    for chunk in chunks:
        events.extend(decoder.feed(chunk))
    events.extend(decoder.close())
    return events


def test_single_event_is_dispatched_on_blank_line() -> None:
    decoder = SSEDecoder()
    events = drain(decoder, [b'data: {"a": 1}\n\n'])

    assert [event.data for event in events] == ['{"a": 1}']
    assert decoder.incomplete_event_discarded is False


def test_event_split_across_every_byte_boundary() -> None:
    payload = b'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
    decoder = SSEDecoder()

    events = drain(
        decoder, [payload[index : index + 1] for index in range(len(payload))]
    )

    assert [event.data for event in events] == [
        '{"choices": [{"delta": {"content": "hi"}}]}'
    ]


def test_multibyte_character_split_across_chunks_is_reassembled() -> None:
    payload = 'data: {"content": "π 世界 🙂"}\n\n'.encode()
    midpoint = payload.index(b"\xcf") + 1  # inside the two-byte pi character

    decoder = SSEDecoder()
    events = drain(decoder, [payload[:midpoint], payload[midpoint:]])

    assert [event.data for event in events] == ['{"content": "π 世界 🙂"}']


def test_crlf_and_lone_cr_terminators_are_supported() -> None:
    decoder = SSEDecoder()
    events = drain(decoder, [b"data: one\r\n\r\n", b"data: two\r\rdata: three\n\n"])

    assert [event.data for event in events] == ["one", "two", "three"]


def test_trailing_cr_waits_for_the_next_chunk() -> None:
    decoder = SSEDecoder()

    first = list(decoder.feed(b"data: value\r"))
    assert first == []

    second = list(decoder.feed(b"\n\n"))
    assert [event.data for event in second] == ["value"]


def test_comments_are_counted_and_never_dispatched() -> None:
    decoder = SSEDecoder()
    events = drain(decoder, [b": keepalive\n\n", b":\n", b"data: real\n\n"])

    assert [event.data for event in events] == ["real"]
    assert decoder.comment_count == 2


def test_multiple_data_lines_are_joined_with_newline() -> None:
    decoder = SSEDecoder()
    events = drain(decoder, [b"data: first\ndata: second\n\n"])

    assert [event.data for event in events] == ["first\nsecond"]


def test_field_value_keeps_all_but_one_leading_space() -> None:
    decoder = SSEDecoder()
    events = drain(decoder, [b"data:  padded\n\n"])

    assert [event.data for event in events] == [" padded"]


def test_event_and_id_fields_are_captured() -> None:
    decoder = SSEDecoder()
    events = drain(decoder, [b"event: chunk\nid: 42\ndata: body\n\n"])

    assert events == [SSEEvent(data="body", event="chunk", last_event_id="42")]


def test_unknown_fields_and_colonless_lines_are_ignored() -> None:
    decoder = SSEDecoder()
    events = drain(decoder, [b"retry: 500\nnonsense\ndata: body\n\n"])

    assert [event.data for event in events] == ["body"]


def test_blank_line_without_data_dispatches_nothing() -> None:
    decoder = SSEDecoder()
    events = drain(decoder, [b"\n\n", b"event: only-type\n\n"])

    assert events == []


def test_unterminated_final_event_is_discarded_and_flagged() -> None:
    """End of stream is not a blank line, so the frame is never dispatched.

    Dispatching it would hand the collector a ``[DONE]`` the provider
    never finished sending, which reads as a clean end of stream.
    """
    decoder = SSEDecoder()
    events = drain(decoder, [b"data: [DONE]"])

    assert events == []
    assert decoder.incomplete_event_discarded is True


def test_close_is_idempotent() -> None:
    decoder = SSEDecoder()
    list(decoder.feed(b"data: one\n\n"))

    assert list(decoder.close()) == []
    assert list(decoder.close()) == []


def test_feeding_a_closed_decoder_is_an_error() -> None:
    decoder = SSEDecoder()
    list(decoder.close())

    with pytest.raises(SSEDecodeError, match="closed"):
        list(decoder.feed(b"data: late\n\n"))


def test_invalid_utf8_raises_a_decode_error() -> None:
    decoder = SSEDecoder()

    with pytest.raises(SSEDecodeError, match="not valid UTF-8"):
        list(decoder.feed(b"data: \xff\xfe\n\n"))


def test_stream_ending_mid_character_raises_a_decode_error() -> None:
    decoder = SSEDecoder()
    list(decoder.feed(b"data: \xcf"))

    with pytest.raises(SSEDecodeError, match="mid-character"):
        list(decoder.close())


# --- Sixth review pass: end of stream and byte order mark --------------------


def test_incomplete_ordinary_event_at_eof_is_discarded() -> None:
    """A frame with no blank line was cut in transit, so it is not an event."""
    decoder = SSEDecoder()
    events = drain(decoder, [b'data: {"choices": [{"delta": {"content": "hi"}}]}\n'])

    assert events == []
    assert decoder.incomplete_event_discarded is True


def test_incomplete_done_after_a_complete_event_is_discarded() -> None:
    decoder = SSEDecoder()
    events = drain(decoder, [b'data: {"a": 1}\n\n', b"data: [DONE]"])

    assert [event.data for event in events] == ['{"a": 1}']
    assert decoder.incomplete_event_discarded is True


@pytest.mark.parametrize("split", [1, 3, 6, 9])
def test_incomplete_done_is_discarded_however_it_is_fragmented(split: int) -> None:
    payload = b"data: [DONE]"
    decoder = SSEDecoder()
    events = drain(decoder, [payload[:split], payload[split:]])

    assert events == []
    assert decoder.incomplete_event_discarded is True


def test_a_terminated_stream_is_not_marked_incomplete() -> None:
    decoder = SSEDecoder()
    events = drain(decoder, [b'data: {"a": 1}\n\n', b"data: [DONE]\n\n"])

    assert [event.data for event in events] == ['{"a": 1}', "[DONE]"]
    assert decoder.incomplete_event_discarded is False


def test_a_complete_trailing_comment_does_not_mark_the_stream_incomplete() -> None:
    decoder = SSEDecoder()
    events = drain(decoder, [b'data: {"a": 1}\n\n', b": keepalive\n"])

    assert [event.data for event in events] == ['{"a": 1}']
    assert decoder.incomplete_event_discarded is False
    assert decoder.comment_count == 1


def test_a_leading_byte_order_mark_is_ignored() -> None:
    """Without this the first field name is ``\ufeffdata`` and the event vanishes."""
    decoder = SSEDecoder()
    events = drain(decoder, ["\ufeff".encode() + b'data: {"a": 1}\n\n'])

    assert [event.data for event in events] == ['{"a": 1}']


@pytest.mark.parametrize("split", [1, 2])
def test_a_byte_order_mark_split_across_chunks_is_ignored(split: int) -> None:
    mark = "\ufeff".encode()
    decoder = SSEDecoder()
    events = drain(decoder, [mark[:split], mark[split:] + b'data: {"a": 1}\n\n'])

    assert [event.data for event in events] == ['{"a": 1}']


def test_only_the_first_byte_order_mark_is_stripped() -> None:
    """A U+FEFF anywhere after the first character is ordinary content."""
    decoder = SSEDecoder()
    events = drain(decoder, ['\ufeffdata: {"a": "\ufeff\ufeff"}\n\n'.encode()])

    assert [event.data for event in events] == ['{"a": "\ufeff\ufeff"}']


def test_a_byte_order_mark_alone_does_not_start_an_event() -> None:
    decoder = SSEDecoder()
    events = drain(decoder, ["\ufeff".encode()])

    assert events == []
    assert decoder.incomplete_event_discarded is False
