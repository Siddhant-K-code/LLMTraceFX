"""Synthetic prompts that keep a trace's prefix-sharing structure.

There is no tokenizer here. One "unit" is one word from a fixed list of 256
short English words, each written with a leading space. The target length is
a word count, so the length a server tokenizes is labelled ``estimated`` and
the provider's own ``prompt_tokens`` is recorded next to it at run time.

Words are drawn with SHAKE-256 rather than ``random`` so the same salt gives
the same bytes on every Python version.

Mooncake: the text of hash block ``h`` depends only on the salt, the trace
sha256 and ``h``. Two requests that share their first ``k`` hash ids share
byte-identical text for those ``k`` blocks and differ from the next one. A
partial last block is the leading words of that id's block. The parser has
already checked that an id always names the same length.

BurstGPT has no prefix information, so each prompt is seeded by its source
line and prompts share no structure.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from collections.abc import Sequence

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


def new_salt() -> str:
    """A fresh salt, so a replay cannot hit cache entries left by an earlier run."""
    return secrets.token_hex(16)


def validate_salt(salt: str) -> str:
    if not _SALT.fullmatch(salt):
        raise PromptSaltError(
            "--prompt-salt must be 1 to 128 characters from A-Z a-z 0-9 . _ : -"
        )
    return salt


def _words(seed: bytes, count: int) -> str:
    if count <= 0:
        return ""
    stream = hashlib.shake_256(seed).digest(count)
    return "".join(" " + WORDS[byte] for byte in stream)


def word_count(text: str) -> int:
    """The estimated prompt length for a text the trace supplies."""
    return len(text.split())


class PromptBuilder:
    """Builds deterministic synthetic prompts for one trace and salt."""

    def __init__(self, *, salt: str, trace_sha256: str, block_size: int | None) -> None:
        self._salt = validate_salt(salt)
        self._trace = trace_sha256
        self._block_size = block_size
        self._blocks: dict[int, str] = {}

    def _seed(self, kind: str, key: int) -> bytes:
        return f"{PROMPT_METHOD}\0{self._salt}\0{self._trace}\0{kind}\0{key}".encode(
            "ascii"
        )

    def block_text(self, hash_id: int) -> str:
        """The full-length text for one Mooncake hash block."""
        if self._block_size is None:
            raise ValueError("hash blocks need a block size")
        text = self._blocks.get(hash_id)
        if text is None:
            text = _words(self._seed("block", hash_id), self._block_size)
            if len(self._blocks) < 65536:
                self._blocks[hash_id] = text
        return text

    def for_hash_ids(self, hash_ids: Sequence[int], input_tokens: int) -> str:
        if self._block_size is None:
            raise ValueError("hash blocks need a block size")
        last = input_tokens - (len(hash_ids) - 1) * self._block_size
        if not hash_ids or not 1 <= last <= self._block_size:
            raise ValueError("hash_ids do not match input_tokens")
        parts = [self.block_text(block) for block in hash_ids[:-1]]
        final = self.block_text(hash_ids[-1])
        if last < self._block_size:
            # Every word is one leading space plus four ASCII letters.
            final = final[: last * 5]
        parts.append(final)
        return "".join(parts)

    def for_line(self, line: int, input_tokens: int) -> str:
        """An unstructured prompt seeded by the source line."""
        return _words(self._seed("line", line), input_tokens)
