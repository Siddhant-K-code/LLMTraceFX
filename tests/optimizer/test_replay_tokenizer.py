import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _tokenizer_fakes import (
    FAKE_SHA,
    ByteLevel,
    CharLevel,
    MergesRepeats,
    Metaspace,
    fake_tokenizer,
)

from llmtracefx.optimizer.replay import tokenizer as tok
from llmtracefx.optimizer.replay.formats import parse_trace
from llmtracefx.optimizer.replay.prompts import WORDS
from llmtracefx.optimizer.replay.schedule import Selection, build_schedule
from llmtracefx.optimizer.replay.tokenizer import (
    TOKEN_PROMPT_METHOD,
    TokenizerError,
    fit_units,
    load_tokenizer,
)

FIXTURES = Path(__file__).parent / "fixtures" / "replay"
MOONCAKE = parse_trace(FIXTURES / "mooncake_small.jsonl", "mooncake")
AIPERF = parse_trace(FIXTURES / "aiperf_single_turn.jsonl", "aiperf-single-turn")
MULTI = parse_trace(FIXTURES / "aiperf_multi_turn.jsonl", "aiperf-multi-turn")


def test_units_keep_only_one_token_words() -> None:
    units = fit_units(fake_tokenizer(ByteLevel()))
    kept = units.vocabulary.words
    assert kept == tuple(w for w in WORDS if "k" not in w)
    assert units.vocabulary.leading_space is True
    assert units.vocabulary.method == TOKEN_PROMPT_METHOD


def test_sentencepiece_style_drops_the_first_space() -> None:
    units = fit_units(fake_tokenizer(Metaspace()))
    assert units.vocabulary.words == WORDS
    assert units.vocabulary.leading_space is False
    schedule = build_schedule(
        MOONCAKE, Selection(), salt="s", tokenizer=fake_tokenizer(Metaspace())
    )
    first = next(schedule.prompts())
    assert not first.startswith(" ")


def test_too_few_one_token_words_is_refused() -> None:
    with pytest.raises(TokenizerError, match=r"only 0 of the 256 .* at least 64"):
        fit_units(fake_tokenizer(CharLevel()))


@pytest.mark.parametrize("encoder", [ByteLevel(), Metaspace()])
def test_mooncake_lengths_and_shared_prefixes_are_exact(encoder: Any) -> None:
    tokenizer = fake_tokenizer(encoder)
    schedule = build_schedule(MOONCAKE, Selection(), salt="s", tokenizer=tokenizer)
    targets = [r.target_input_tokens for r in schedule.requests]
    assert [r.tokenizer_prompt_tokens for r in schedule.requests] == targets
    assert [r.estimated_prompt_tokens for r in schedule.requests] == targets

    prompts = list(schedule.prompts())
    ids = encoder.encode_batch(prompts)
    assert [len(i) for i in ids] == targets
    # Request 3 is the first two 512-token blocks that requests 0 and 1 share.
    for shared in (0, 1):
        assert prompts[shared].startswith(prompts[3])
        assert ids[shared][:1024] == ids[3]


def test_tokenizer_text_differs_from_word_mode() -> None:
    plain = build_schedule(MOONCAKE, Selection(), salt="s")
    exact = build_schedule(
        MOONCAKE, Selection(), salt="s", tokenizer=fake_tokenizer(ByteLevel())
    )
    assert plain.requests[0].prompt_sha256 != exact.requests[0].prompt_sha256
    assert plain.requests[0].tokenizer_prompt_tokens is None


def test_a_merge_refuses_the_schedule() -> None:
    with pytest.raises(TokenizerError, match=r"line \d+: .*adjacent words merged"):
        build_schedule(
            MOONCAKE, Selection(), salt="s", tokenizer=fake_tokenizer(MergesRepeats())
        )


@pytest.mark.parametrize("encoder", [ByteLevel(), Metaspace()])
def test_multi_turn_counts_every_message_exactly(encoder: Any) -> None:
    schedule = build_schedule(
        MULTI, Selection(), salt="s", tokenizer=fake_tokenizer(encoder)
    )
    builder = schedule.prompt_builder()
    for index, request in enumerate(schedule.requests):
        history = schedule.history_at(builder, index)
        texts = [text for _, text in history] + [schedule.entries[index].text or ""]
        counted = sum(len(ids) for ids in encoder.encode_batch(texts))
        assert request.tokenizer_prompt_tokens == counted
        replies = [text for role, text in history if role == "assistant"]
        earlier = schedule.entries[index - len(replies) : index]
        assert [len(ids) for ids in encoder.encode_batch(replies)] == [
            e.output_tokens for e in earlier
        ]
    assert max(r.history_turns or 0 for r in schedule.requests) == 2
    history_record = schedule.to_document()["prompts"]["history"]
    assert history_record["assistant_method"] == TOKEN_PROMPT_METHOD + ":reply"
    assert history_record["assistant_token_provenance"] == "tokenizer_exact"
    assert "chat template" in history_record["counted"]


class _GrowsLongText(ByteLevel):
    """One token per word in the probes, but texts of 3+ words gain a token."""

    def pieces(self, text: str) -> list[str]:
        pieces = super().pieces(text)
        return pieces + ["!"] if len(pieces) >= 3 else pieces


def test_multi_turn_refuses_a_reply_of_inexact_length() -> None:
    # User turns are trace text and only counted; the synthetic replies of
    # 3 or more words no longer encode one token per word.
    with pytest.raises(TokenizerError, match=r"line \d+: .*adjacent words merged"):
        build_schedule(
            MULTI, Selection(), salt="s", tokenizer=fake_tokenizer(_GrowsLongText())
        )


def test_trace_text_is_counted_as_it_encodes() -> None:
    schedule = build_schedule(
        AIPERF, Selection(), salt="s", tokenizer=fake_tokenizer(ByteLevel())
    )
    counts = [r.tokenizer_prompt_tokens for r in schedule.requests]
    assert counts == [len(ByteLevel().pieces(e.text or "")) for e in AIPERF.entries]
    assert counts[0] == 4


def test_schedule_records_the_tokenizer_and_what_is_counted() -> None:
    schedule = build_schedule(
        MOONCAKE, Selection(), salt="s", tokenizer=fake_tokenizer(ByteLevel())
    )
    document = schedule.to_document()
    prompts = document["prompts"]
    assert prompts["method"] == TOKEN_PROMPT_METHOD
    assert prompts["unit"] == "token"
    assert prompts["length_provenance"] == "tokenizer_exact"
    assert prompts["tokenizer"]["file_sha256"] == FAKE_SHA
    assert prompts["tokenizer"]["commit"] == "c" * 40
    assert prompts["tokenizer"]["chat_template"] == "not counted"
    assert prompts["unit_words"] == len(schedule.vocabulary.words)
    summary = document["summary"]
    assert summary["tokenizer_prompt_tokens_total"] == 4024
    assert summary["tokenizer_provenance"] == "tokenizer_exact"
    assert document["requests"][0]["tokenizer_prompt_tokens"] == 1100


def test_word_mode_document_has_no_tokenizer_fields() -> None:
    document = build_schedule(MOONCAKE, Selection(), salt="s").to_document()
    assert "tokenizer" not in document["prompts"]
    assert "tokenizer_prompt_tokens_total" not in document["summary"]


# Loading, with fake modules in place of tokenizers and huggingface_hub.


class _Encoding:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids


class _FakeTokenizer:
    def __init__(self, text: str) -> None:
        self.config = json.loads(text)
        self.calls: list[bool] = []

    def encode_batch(
        self, texts: list[str], add_special_tokens: bool = True
    ) -> list[_Encoding]:
        self.calls.append(add_special_tokens)
        return [_Encoding(ids) for ids in ByteLevel().encode_batch(texts)]


@pytest.fixture
def fake_modules(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    downloads: list[dict[str, Any]] = []

    def hf_hub_download(**kwargs: Any) -> str:
        downloads.append(kwargs)
        if kwargs["repo_id"] == "org/missing":
            raise RuntimeError("404 Repository Not Found\nmore detail")
        folder = tmp_path / "hub" / "snapshots" / ("a" * 40)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "tokenizer.json").write_text('{"model": "hub"}')
        return str(folder / "tokenizer.json")

    def from_str(text: str) -> _FakeTokenizer:
        return _FakeTokenizer(text)

    modules = {
        "tokenizers": SimpleNamespace(Tokenizer=SimpleNamespace(from_str=from_str)),
        "huggingface_hub": SimpleNamespace(hf_hub_download=hf_hub_download),
    }
    monkeypatch.setattr(tok, "_import", modules.__getitem__)
    return {"downloads": downloads}


def test_load_a_local_file_and_directory(
    fake_modules: dict[str, Any], tmp_path: Path
) -> None:
    folder = tmp_path / "secret-folder-name"
    folder.mkdir()
    path = folder / "tokenizer.json"
    path.write_text('{"model": "local"}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    for spec in (str(path), str(folder)):
        loaded = load_tokenizer(spec)
        assert loaded.identity.source == "local_file"
        assert loaded.identity.name == "tokenizer.json"
        assert loaded.identity.file_sha256 == digest
        assert "secret-folder-name" not in json.dumps(loaded.identity.to_dict())
        loaded.encoder.encode_batch(["a b"])
        assert loaded.encoder._tokenizer.calls == [False]  # type: ignore[attr-defined]
    assert tok.display_name(str(folder)) == "secret-folder-name"
    assert tok.display_name("org/model") == "org/model"
    assert fake_modules["downloads"] == []

    with pytest.raises(TokenizerError, match="--tokenizer-revision applies"):
        load_tokenizer(str(path), "main")
    with pytest.raises(TokenizerError, match="directory has no tokenizer.json"):
        load_tokenizer(str(tmp_path))
    bad = tmp_path / "bad.json"
    bad.write_text("not json")
    with pytest.raises(TokenizerError, match="bad.json is not a valid"):
        load_tokenizer(str(bad))


def test_load_a_repo_id_records_the_commit(fake_modules: dict[str, Any]) -> None:
    loaded = load_tokenizer("org/model", "v1")
    identity = loaded.identity.to_dict()
    assert identity["source"] == "huggingface_hub"
    assert identity["name"] == "org/model"
    assert identity["revision"] == "v1"
    assert identity["commit"] == "a" * 40
    assert fake_modules["downloads"] == [
        {"repo_id": "org/model", "filename": "tokenizer.json", "revision": "v1"}
    ]
    with pytest.raises(
        TokenizerError, match=r"org/missing: 404 Repository Not Found \("
    ):
        load_tokenizer("org/missing")
    with pytest.raises(TokenizerError, match="is not a file, a directory"):
        load_tokenizer("not a repo id")


def test_missing_package_is_refused_with_an_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    with pytest.raises(
        TokenizerError,
        match=r"'tokenizers' package.*uv tool install 'llmtracefx\[tokenizer\]'",
    ):
        load_tokenizer("org/model")


@pytest.mark.parametrize("encoder", [ByteLevel(), Metaspace()])
@pytest.mark.parametrize(
    ("name", "fmt"),
    [("weka_small.jsonl", "weka"), ("otel_genai_small.jsonl", "otel-genai")],
)
def test_session_trace_lengths_are_exact(encoder: Any, name: str, fmt: str) -> None:
    trace = parse_trace(FIXTURES / name, fmt)
    schedule = build_schedule(
        trace, Selection(), salt="s", tokenizer=fake_tokenizer(encoder)
    )
    targets = [r.target_input_tokens for r in schedule.requests]
    assert [r.tokenizer_prompt_tokens for r in schedule.requests] == targets
    ids = encoder.encode_batch(list(schedule.prompts()))
    assert [len(i) for i in ids] == targets
    assert schedule.to_document()["prompts"]["length_provenance"] == "tokenizer_exact"
