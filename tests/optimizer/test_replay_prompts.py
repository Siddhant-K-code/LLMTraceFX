import pytest

from llmtracefx.optimizer.replay.prompts import (
    WORDS,
    PromptBuilder,
    PromptSaltError,
    new_salt,
    validate_salt,
    word_count,
)

TRACE = "0" * 64


def _builder(salt: str = "s1", block_size: int | None = 8) -> PromptBuilder:
    return PromptBuilder(salt=salt, trace_sha256=TRACE, block_size=block_size)


def test_shared_hash_ids_give_byte_identical_prefixes() -> None:
    builder = _builder()
    first = builder.for_hash_ids([10, 11, 12], 24)
    second = builder.for_hash_ids([10, 11, 99], 24)
    unrelated = builder.for_hash_ids([50, 11, 12], 24)
    # Every word is five characters, so 8 words per block is 40 characters.
    assert first[:80] == second[:80]
    assert first[80:] != second[80:]
    assert first[:40] != unrelated[:40]
    assert word_count(first) == 24


def test_partial_last_block_is_a_prefix_of_the_full_block() -> None:
    builder = _builder()
    partial = builder.for_hash_ids([10, 11], 11)
    assert word_count(partial) == 11
    assert partial[40:] == builder.block_text(11)[:15]
    assert builder.for_hash_ids([11], 8) == builder.block_text(11)


def test_salt_and_trace_select_the_text() -> None:
    assert _builder("s1").block_text(3) == _builder("s1").block_text(3)
    assert _builder("s1").block_text(3) != _builder("s2").block_text(3)
    other_trace = PromptBuilder(salt="s1", trace_sha256="1" * 64, block_size=8)
    assert other_trace.block_text(3) != _builder("s1").block_text(3)


def test_line_prompts_use_the_word_list() -> None:
    text = _builder(block_size=None).for_line(7, 50)
    assert word_count(text) == 50
    assert set(text.split()) <= set(WORDS)
    assert text.startswith(" ")
    with pytest.raises(ValueError, match="block size"):
        _builder(block_size=None).block_text(1)


def test_hash_ids_must_match_the_length() -> None:
    with pytest.raises(ValueError, match="do not match"):
        _builder().for_hash_ids([1, 2], 20)


@pytest.mark.parametrize("salt", ["", "has space", "x" * 129, "semi;colon"])
def test_bad_salts_are_refused(salt: str) -> None:
    with pytest.raises(PromptSaltError):
        validate_salt(salt)


def test_new_salt_is_valid_and_fresh() -> None:
    assert validate_salt(new_salt()) != new_salt()
