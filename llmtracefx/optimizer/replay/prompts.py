"""Synthetic prompts that keep a trace's prefix-sharing structure.

One "unit" is one word from a fixed list of 256 short English words, each
written with a leading space. The target length is a word count, so the
length a server tokenizes is labelled ``estimated`` and the provider's own
``prompt_tokens`` is recorded next to it at run time. With ``--tokenizer``
the words are narrowed to those that are one token each and every prompt is
checked against that tokenizer; see ``tokenizer.py``.

Words are drawn with SHAKE-256 rather than ``random`` so the same salt gives
the same bytes on every Python version.

Mooncake: the text of hash block ``h`` depends only on the salt, the trace
sha256 and ``h``. Two requests that share their first ``k`` hash ids share
byte-identical text for those ``k`` blocks and differ from the next one. A
partial last block is the leading words of that id's block. The parser has
already checked that an id always names the same length.

Weka: the same as Mooncake, except that ids are numbered per conversation,
so a block's seed also holds its conversation's source line. A request whose
input length is past its last full hashed block gets a tail seeded by the
request itself, so the tail is never shared.

OTel GenAI spans carry no prefix information, so each prompt is seeded by its
span's place in the file and prompts share no structure.

BurstGPT has no prefix information, so each prompt is seeded by its source
line and prompts share no structure.

Multi-turn: a later turn resends the earlier turns. The user turns are the
trace's own text. No response text is ever stored, so each earlier assistant
turn is a synthetic stand-in whose word count is that turn's declared
``output_length``, seeded by the turn's place in the file. The same turn
therefore has the same bytes in every later request of its session, which
keeps the conversation prefix shared as it grows.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass

WORDS: tuple[str, ...] = tuple("""
    able acid aged also area army away baby back ball band bank base bath bear
    beat been beer bell belt best bird blow blue boat body bone book boot born
    boss both bowl bulk burn bush busy cake call calm came camp card care case
    cash cast cell chat chip city club coal coat code cold come cook cool cope
    copy core cost crew crop dark data date dawn days dead deal dean dear debt
    deep deny desk diet disc disk does done door dose down draw drew drop drug
    dual duke dust duty each earn ease east easy edge else even ever evil exit
    face fact fail fair fall farm fast fate fear feed feel feet fell felt file
    fill film find fine fire firm fish five flat flow food foot ford form fort
    four free from fuel full fund gain game gate gave gear gene gift girl give
    glad goal goes gold golf gone good gray grew grey grow gulf hair half hall
    hand hang hard harm hate have head hear heat held hell help here hero high
    hill hire hold hole holy home hope host hour huge hung hunt hurt idea inch
    into iron item jack jane jean john join jump jury just keen keep kent kept
    kick kill kind king knee knew know lack lady laid lake land lane last late
    lead left less life lift like line link list live load loan lock logo long
    look lord lose loss lost loud love luck made mail main make male many mark mass
    """.split())
if len(WORDS) != 256 or len(set(WORDS)) != 256 or {len(w) for w in WORDS} != {4}:
    raise RuntimeError("the replay word list must hold 256 distinct 4-letter words")

WORD_LIST_SHA256 = hashlib.sha256(" ".join(WORDS).encode("ascii")).hexdigest()
PROMPT_METHOD = "llmtracefx.replay.shake256_words.v1"
_SALT = re.compile(r"[A-Za-z0-9._:-]{1,128}")


class PromptSaltError(ValueError):
    pass


@dataclass(frozen=True)
class Vocabulary:
    """The words a prompt is drawn from and how its first word is written."""

    words: tuple[str, ...] = WORDS
    leading_space: bool = True
    method: str = PROMPT_METHOD

    def words_sha256(self) -> str:
        return hashlib.sha256(" ".join(self.words).encode("ascii")).hexdigest()


WORD_VOCABULARY = Vocabulary()


def new_salt() -> str:
    """A fresh salt, so a replay cannot hit cache entries left by an earlier run."""
    return secrets.token_hex(16)


def validate_salt(salt: str) -> str:
    if not _SALT.fullmatch(salt):
        raise PromptSaltError(
            "--prompt-salt must be 1 to 128 characters from A-Z a-z 0-9 . _ : -"
        )
    return salt


def _words(seed: bytes, count: int, words: tuple[str, ...]) -> str:
    if count <= 0:
        return ""
    stream = hashlib.shake_256(seed).digest(count)
    size = len(words)
    return "".join(" " + words[byte % size] for byte in stream)


def word_count(text: str) -> int:
    """The estimated prompt length for a text the trace supplies."""
    return len(text.split())


class PromptBuilder:
    """Builds deterministic synthetic prompts for one trace and salt."""

    def __init__(
        self,
        *,
        salt: str,
        trace_sha256: str,
        block_size: int | None,
        vocabulary: Vocabulary = WORD_VOCABULARY,
    ) -> None:
        self._salt = validate_salt(salt)
        self._trace = trace_sha256
        self._block_size = block_size
        self._vocabulary = vocabulary
        self._blocks: dict[tuple[int | None, int], str] = {}

    def _seed(self, kind: str, key: int | str) -> bytes:
        method = self._vocabulary.method
        return f"{method}\0{self._salt}\0{self._trace}\0{kind}\0{key}".encode("ascii")

    def _finish(self, text: str) -> str:
        # Every prompt starts the same way, so dropping the first space keeps
        # shared prefixes byte-identical.
        return text if self._vocabulary.leading_space else text[1:]

    def block_text(self, hash_id: int, scope: int | None = None) -> str:
        """The full-length text for one hash block.

        ``scope`` is the conversation a Weka id belongs to; a Mooncake id is
        global and has none.
        """
        if self._block_size is None:
            raise ValueError("hash blocks need a block size")
        text = self._blocks.get((scope, hash_id))
        if text is None:
            key: int | str = hash_id if scope is None else f"{scope}.{hash_id}"
            text = _words(
                self._seed("block", key), self._block_size, self._vocabulary.words
            )
            if len(self._blocks) < 65536:
                self._blocks[(scope, hash_id)] = text
        return text

    def for_hash_ids(
        self,
        hash_ids: Sequence[int],
        input_tokens: int,
        *,
        scope: int | None = None,
        tail_key: str | None = None,
    ) -> str:
        """A prompt made of hash blocks.

        With ``tail_key``, tokens past the last full block are words seeded
        by that key, for a trace that hashes full blocks only.
        """
        if self._block_size is None:
            raise ValueError("hash blocks need a block size")
        hashed = len(hash_ids) * self._block_size
        if tail_key is not None and hashed <= input_tokens:
            blocks = "".join(self.block_text(block, scope) for block in hash_ids)
            tail = _words(
                self._seed("tail", tail_key),
                input_tokens - hashed,
                self._vocabulary.words,
            )
            return self._finish(blocks + tail)
        last = input_tokens - (len(hash_ids) - 1) * self._block_size
        if not hash_ids or not 1 <= last <= self._block_size:
            raise ValueError("hash_ids do not match input_tokens")
        parts = [self.block_text(block, scope) for block in hash_ids[:-1]]
        final = self.block_text(hash_ids[-1], scope)
        if last < self._block_size:
            # Every word is one leading space plus four ASCII letters.
            final = final[: last * 5]
        parts.append(final)
        return self._finish("".join(parts))

    def for_line(self, line: int, input_tokens: int) -> str:
        """An unstructured prompt seeded by the source line."""
        return self._finish(
            _words(self._seed("line", line), input_tokens, self._vocabulary.words)
        )

    def for_item(self, line: int, position: int, input_tokens: int) -> str:
        """An unstructured prompt seeded by one item of a source line."""
        return self._finish(
            _words(
                self._seed("item", f"{line}.{position}"),
                input_tokens,
                self._vocabulary.words,
            )
        )

    def for_reply(self, line: int, line_turn: int, words: int) -> str:
        """A synthetic stand-in for the assistant's answer to one trace turn."""
        return self._finish(
            _words(
                self._seed("reply", f"{line}.{line_turn}"),
                words,
                self._vocabulary.words,
            )
        )
