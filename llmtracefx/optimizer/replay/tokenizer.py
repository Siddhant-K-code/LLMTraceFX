"""An optional tokenizer that makes synthetic prompt lengths exact.

Without a tokenizer a prompt length is a word count and is labelled
``estimated``. With ``--tokenizer`` the prompt is still built from the replay
word list, but only from words that are one token each in that tokenizer, and
every prompt is encoded and checked token by token before anything is
written or sent:

* the prompt must encode to exactly one token per word, with the token ids
  the probe found for those words. Any merge across two words, whether inside
  a Mooncake block or across a block boundary, changes the ids and refuses
  the schedule. Nothing is padded or trimmed to make a count fit;
* so the prompt is exactly ``input_length`` tokens, and the first ``k`` hash
  blocks are exactly the first ``k * block_size`` tokens. Requests that share
  leading hash ids share byte-identical text and identical token ids for it.

What is counted is the prompt text alone, encoded without special tokens.
The chat template and the special tokens a server adds are not counted: the
template belongs to the server and this client cannot see which one it
uses. The difference shows up as provider ``prompt_tokens`` minus the
tokenizer count in the result.

The ``tokenizers`` package is an optional extra. It is imported only when
``--tokenizer`` is given, and its absence is refused with an install hint.
"""

from __future__ import annotations

import hashlib
import importlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from ...extras import extra_install_hint
from .prompts import WORDS, Vocabulary

TOKEN_PROMPT_METHOD = "llmtracefx.replay.shake256_tokens.v1"
MIN_UNIT_WORDS = 64
TOKENIZER_FILE = "tokenizer.json"
INSTALL_HINT = (
    f"install the optional extra with {extra_install_hint('tokenizer')}"
)
# A probe prefix, so the word under test is not the first thing in the text.
_ANCHOR = "."
_REPO_ID = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}(/[A-Za-z0-9][A-Za-z0-9._-]{0,95})?"
)
_COMMIT = re.compile(r"[0-9a-f]{40}")


class TokenizerError(ValueError):
    pass


class Encoder(Protocol):
    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Token ids for each text, with no special tokens added."""
        ...


@dataclass(frozen=True)
class TokenizerIdentity:
    source: str
    """``local_file`` or ``huggingface_hub``."""
    name: str
    """The repo id, or the file name of a local tokenizer.json."""
    file_sha256: str
    revision: str | None = None
    commit: str | None = None
    library_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "name": self.name,
            "file": TOKENIZER_FILE,
            "file_sha256": self.file_sha256,
            "revision": self.revision,
            "commit": self.commit,
            "library": "tokenizers",
            "library_version": self.library_version,
            "special_tokens": "not added",
            "chat_template": "not counted",
        }


@dataclass(frozen=True)
class ReplayTokenizer:
    identity: TokenizerIdentity
    encoder: Encoder


@dataclass(frozen=True)
class TokenUnits:
    """The words that are one token each, and the ids they encode to."""

    vocabulary: Vocabulary
    word_ids: dict[str, int]
    first_word_ids: dict[str, int]

    def expected_ids(self, words: Sequence[str]) -> list[int]:
        if not words:
            return []
        return [self.first_word_ids[words[0]]] + [self.word_ids[w] for w in words[1:]]


class _TokenizersEncoder:
    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        encodings = self._tokenizer.encode_batch(list(texts), add_special_tokens=False)
        return [list(encoding.ids) for encoding in encodings]


def _import(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise TokenizerError(
            f"--tokenizer needs the '{module}' package, which is not installed; "
            f"{INSTALL_HINT}"
        ) from exc


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def display_name(spec: str) -> str:
    """What to record for ``--tokenizer``: a file name, never a local path."""
    path = Path(spec).expanduser()
    if path.exists():
        return path.name
    return spec


def _local_file(spec: str) -> Path | None:
    path = Path(spec).expanduser()
    if path.is_dir():
        path = path / TOKENIZER_FILE
        if not path.is_file():
            raise TokenizerError(f"--tokenizer directory has no {TOKENIZER_FILE}")
        return path
    if path.exists():
        if not path.is_file():
            raise TokenizerError(
                "--tokenizer must name a file, a directory or a repo id"
            )
        return path
    return None


def _download(repo_id: str, revision: str | None) -> tuple[Path, str | None]:
    hub = _import("huggingface_hub")
    try:
        downloaded = Path(
            hub.hf_hub_download(
                repo_id=repo_id, filename=TOKENIZER_FILE, revision=revision
            )
        )
    except Exception as exc:  # the hub raises many exception types
        reason = (str(exc).strip().splitlines() or [type(exc).__name__])[0]
        raise TokenizerError(
            f"could not fetch {TOKENIZER_FILE} from {repo_id}"
            f"{'' if revision is None else '@' + revision}: {reason} (only fast "
            f"tokenizers with a {TOKENIZER_FILE} are supported; gated repos need "
            "a Hugging Face login)"
        ) from exc
    # The hub cache keeps files under snapshots/<commit>/.
    commit = downloaded.parent.name
    return downloaded, commit if _COMMIT.fullmatch(commit) else None


def load_tokenizer(spec: str, revision: str | None = None) -> ReplayTokenizer:
    """Load a tokenizer.json from a local path or a Hugging Face repo id."""
    tokenizers = _import("tokenizers")
    local = _local_file(spec)
    if local is not None:
        if revision is not None:
            raise TokenizerError(
                "--tokenizer-revision applies to Hugging Face repo ids, not files"
            )
        path, source, name, commit = local, "local_file", local.name, None
    else:
        if not _REPO_ID.fullmatch(spec):
            raise TokenizerError(
                f"--tokenizer {spec!r} is not a file, a directory with "
                f"{TOKENIZER_FILE}, or a Hugging Face repo id"
            )
        path, commit = _download(spec, revision)
        source, name = "huggingface_hub", spec
    try:
        data = path.read_bytes()
        tokenizer = tokenizers.Tokenizer.from_str(data.decode("utf-8"))
    except OSError as exc:
        raise TokenizerError(f"could not read {TOKENIZER_FILE}: {exc}") from exc
    except Exception as exc:  # the Rust binding raises plain Exception
        raise TokenizerError(f"{name} is not a valid {TOKENIZER_FILE}: {exc}") from exc
    identity = TokenizerIdentity(
        source=source,
        name=name,
        file_sha256=hashlib.sha256(data).hexdigest(),
        revision=revision if local is None else None,
        commit=commit,
        library_version=_version("tokenizers"),
    )
    return ReplayTokenizer(identity=identity, encoder=_TokenizersEncoder(tokenizer))


def fit_units(tokenizer: ReplayTokenizer) -> TokenUnits:
    """Keep the replay words that are one token each in this tokenizer.

    A word qualifies when ``" " + word`` after other text adds exactly one
    token. The first word of a prompt has nothing before it, so it is probed
    on its own: with its leading space when that is still one token (byte
    level BPE), otherwise without it (SentencePiece style tokenizers that add
    their own leading space marker). The form that keeps at least
    ``MIN_UNIT_WORDS`` words is used; if neither does, the tokenizer is
    refused.
    """
    encoder = tokenizer.encoder
    name = tokenizer.identity.name
    anchor = encoder.encode_batch([_ANCHOR])[0]
    probes = encoder.encode_batch(
        [_ANCHOR + " " + w for w in WORDS] + [" " + w for w in WORDS] + list(WORDS)
    )
    count = len(WORDS)
    word_ids: dict[str, int] = {}
    for word, ids in zip(WORDS, probes[:count], strict=True):
        if len(ids) == len(anchor) + 1 and ids[: len(anchor)] == anchor:
            word_ids[word] = ids[-1]
    spaced = {
        w: ids[0]
        for w, ids in zip(WORDS, probes[count : 2 * count], strict=True)
        if w in word_ids and ids == [word_ids[w]]
    }
    bare = {
        w: ids[0]
        for w, ids in zip(WORDS, probes[2 * count :], strict=True)
        if w in word_ids and len(ids) == 1
    }
    for leading_space, first in ((True, spaced), (False, bare)):
        if len(first) >= MIN_UNIT_WORDS:
            words = tuple(w for w in WORDS if w in first)
            return TokenUnits(
                vocabulary=Vocabulary(
                    words=words, leading_space=leading_space, method=TOKEN_PROMPT_METHOD
                ),
                word_ids={w: word_ids[w] for w in words},
                first_word_ids=first,
            )
    raise TokenizerError(
        f"tokenizer {name}: only {max(len(spaced), len(bare))} of the "
        f"{count} replay words are one token each, and exact lengths need at "
        f"least {MIN_UNIT_WORDS}"
    )


def count_tokens(
    tokenizer: ReplayTokenizer,
    units: TokenUnits,
    prompts: Sequence[str],
    synthetic: Sequence[bool],
    lines: Sequence[int],
) -> list[int]:
    """Exact token counts for a batch of prompts, checking synthetic ones.

    A synthetic prompt must encode to exactly the ids of its words. Text
    that comes from the trace is counted as it encodes.
    """
    counts: list[int] = []
    encoded = tokenizer.encoder.encode_batch(prompts)
    for prompt, ids, built, line in zip(
        prompts, encoded, synthetic, lines, strict=True
    ):
        if built and ids != units.expected_ids(prompt.split()):
            raise TokenizerError(
                f"line {line}: the synthetic prompt does not encode to one token "
                f"per word in {tokenizer.identity.name} (adjacent words merged "
                "or split), so its length would not be exact; refusing rather "
                "than send a prompt of unknown length"
            )
        counts.append(len(ids))
    return counts
