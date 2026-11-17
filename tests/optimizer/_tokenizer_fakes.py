"""Fake tokenizers for replay tests, so the optional package is never needed."""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence

from llmtracefx.optimizer.replay.tokenizer import ReplayTokenizer, TokenizerIdentity

FAKE_SHA = "f" * 64


def _ids(pieces: Sequence[str]) -> list[int]:
    return [zlib.crc32(piece.encode()) for piece in pieces]


class ByteLevel:
    """Each ``" word"`` is one token, except words with a ``k`` (two tokens)."""

    def pieces(self, text: str) -> list[str]:
        pieces: list[str] = []
        for piece in re.findall(r" ?[^ ]+", text):
            cut = piece.find("k")
            pieces += [piece[:cut], piece[cut:]] if cut > 0 else [piece]
        return pieces

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [_ids(self.pieces(text)) for text in texts]


class Metaspace:
    """SentencePiece style: a leading marker is added and spaces become it."""

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [
            _ids(re.findall(r"\u2581[^\u2581]*", "\u2581" + t.replace(" ", "\u2581")))
            for t in texts
        ]


class MergesRepeats(ByteLevel):
    """Like ByteLevel, but a word that repeats right away merges into one token."""

    def pieces(self, text: str) -> list[str]:
        merged: list[str] = []
        for piece in super().pieces(text):
            if merged and merged[-1].strip() == piece.strip():
                merged[-1] += piece
            else:
                merged.append(piece)
        return merged


class CharLevel:
    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        return [_ids(list(text)) for text in texts]


def fake_tokenizer(encoder: object, name: str = "fake/tokenizer") -> ReplayTokenizer:
    identity = TokenizerIdentity(
        source="huggingface_hub",
        name=name,
        file_sha256=FAKE_SHA,
        revision=None,
        commit="c" * 40,
        library_version="0.0-fake",
    )
    return ReplayTokenizer(identity=identity, encoder=encoder)  # type: ignore[arg-type]
