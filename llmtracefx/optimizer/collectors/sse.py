"""Incremental Server-Sent Events (SSE) decoding for streaming HTTP bodies.

Streaming chat-completions endpoints return ``text/event-stream``. The
network delivers arbitrary byte chunks, so an event, a line, or even a
single UTF-8 code point can be split across two reads. This decoder is
therefore fed raw ``bytes`` and yields only whole, dispatched events.

Framing follows the WHATWG event-stream rules that matter here:

* lines end with ``\\r\\n``, ``\\n`` or a lone ``\\r``;
* a line starting with ``:`` is a comment (providers use these as
  keepalives) and never contributes to an event;
* a field line is split on the first ``:``, with one optional leading
  space removed from the value;
* ``data`` fields accumulate and are joined with ``\\n``;
* a blank line dispatches the buffered event.

Nothing here interprets the payload. JSON parsing, ``[DONE]`` handling
and provider semantics belong to the collector so that a malformed
payload is reported as collector evidence rather than swallowed here.
"""

from __future__ import annotations

import codecs
from collections.abc import Iterator
from dataclasses import dataclass, field


class SSEDecodeError(ValueError):
    """Raised when a byte stream cannot be decoded as UTF-8 event-stream text."""


@dataclass(frozen=True)
class SSEEvent:
    """One dispatched event: its ``data`` payload and framing metadata."""

    data: str
    event: str | None = None
    last_event_id: str | None = None


@dataclass
class SSEDecoder:
    """Feed bytes in, get whole events out.

    The decoder is stateful and single-use per response body. It keeps a
    UTF-8 incremental decoder so multi-byte characters split across two
    network chunks are reassembled instead of raising or producing
    replacement characters.
    """

    comment_count: int = 0
    """Number of ``:``-prefixed comment/keepalive lines seen so far."""

    dispatched_unterminated_event: bool = False
    """True when ``close()`` had to flush an event with no trailing blank line."""

    _decoder: codecs.IncrementalDecoder = field(
        default_factory=lambda: codecs.getincrementaldecoder("utf-8")(errors="strict"),
        repr=False,
    )
    _buffer: str = field(default="", repr=False)
    _data_lines: list[str] = field(default_factory=list, repr=False)
    _event_type: str | None = field(default=None, repr=False)
    _last_event_id: str | None = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)

    def feed(self, chunk: bytes) -> Iterator[SSEEvent]:
        """Decode ``chunk`` and yield every event completed by it."""
        if self._closed:
            raise SSEDecodeError("cannot feed a closed SSE decoder")
        try:
            self._buffer += self._decoder.decode(chunk, False)
        except UnicodeDecodeError as exc:
            raise SSEDecodeError(f"stream is not valid UTF-8: {exc}") from exc
        yield from self._drain_complete_lines()

    def close(self) -> Iterator[SSEEvent]:
        """Finish decoding and yield any event still buffered.

        A well-behaved server ends the last event with a blank line. When
        it does not, the buffered event is still dispatched (the payload
        is complete as far as this layer can tell) and
        ``dispatched_unterminated_event`` records that the framing was
        irregular, so the collector can persist that fact instead of
        hiding it.
        """
        if self._closed:
            return
        self._closed = True
        try:
            self._buffer += self._decoder.decode(b"", True)
        except UnicodeDecodeError as exc:
            raise SSEDecodeError(
                f"stream ended mid-character and is not valid UTF-8: {exc}"
            ) from exc

        yield from self._drain_complete_lines()
        if self._buffer:
            trailing = self._buffer
            self._buffer = ""
            self._consume_line(trailing)
        if self._data_lines:
            self.dispatched_unterminated_event = True
            event = self._build_event()
            if event is not None:
                yield event

    def _drain_complete_lines(self) -> Iterator[SSEEvent]:
        while True:
            line, separator, remainder = _split_first_line(self._buffer)
            if separator is None:
                return
            self._buffer = remainder
            if line == "":
                event = self._build_event()
                if event is not None:
                    yield event
                continue
            self._consume_line(line)

    def _consume_line(self, line: str) -> None:
        if line.startswith(":"):
            self.comment_count += 1
            return
        name, separator, value = line.partition(":")
        if separator and value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data_lines.append(value)
        elif name == "event":
            self._event_type = value
        elif name == "id" and "\x00" not in value:
            self._last_event_id = value
        # Unknown fields (including "retry" and colon-less lines) are
        # ignored, exactly as the event-stream rules require.

    def _build_event(self) -> SSEEvent | None:
        if not self._data_lines:
            self._event_type = None
            return None
        event = SSEEvent(
            data="\n".join(self._data_lines),
            event=self._event_type,
            last_event_id=self._last_event_id,
        )
        self._data_lines = []
        self._event_type = None
        return event


def _split_first_line(buffer: str) -> tuple[str, str | None, str]:
    """Split ``buffer`` at its first complete line terminator.

    Returns ``(line, separator, remainder)``. ``separator`` is ``None``
    when the buffer holds no complete line yet. A trailing lone ``\\r`` is
    treated as incomplete because the next chunk may start with ``\\n``.
    """
    carriage = buffer.find("\r")
    newline = buffer.find("\n")
    if carriage == -1 and newline == -1:
        return buffer, None, ""
    if carriage != -1 and (newline == -1 or carriage < newline):
        if carriage == len(buffer) - 1:
            return buffer, None, ""
        width = 2 if buffer[carriage + 1] == "\n" else 1
        return (
            buffer[:carriage],
            buffer[carriage : carriage + width],
            buffer[carriage + width :],
        )
    return buffer[:newline], "\n", buffer[newline + 1 :]
