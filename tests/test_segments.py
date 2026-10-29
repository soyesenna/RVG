"""K8-1 R1 (`units-greedy/1`): segments cover a revision without gap or overlap, every unit lies
in exactly one segment, every segment's embedding input fits the limit, a revision that fits is
one segment whose input is today's `embedding_input` — and whatever cannot be segmented without
cutting is refused, never truncated."""

import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from rvg.errors import ContractError
from rvg.storage.identity import embedding_input
from rvg.storage.segments import LIMIT_BYTES, segment_revision
from rvg.temporal.models import TemporalDocument

WORDS = ["alpha", "beta", "gamma", "δέλτα", "예산", "項目", "42", "2025"]


def document(sentences, *, title="Report", gap=" "):
    text, units, cursor = "", [], 0
    for n, s in enumerate(sentences):
        if n:
            text += gap
            cursor += len(gap)
        units.append({"id": f"u{n}", "start": cursor, "end": cursor + len(s)})
        text += s
        cursor += len(s)
    return TemporalDocument.model_validate(
        {
            "id": "d",
            "family": "d",
            "revision": 1,
            "title": title,
            "text": text,
            "source": "fixture://d",
            "published_at": "2026-09-01T00:00:00Z",
            "observed_at": "2026-09-01T00:00:00Z",
            "units": units,
        }
    )


# at most 16 x 150 words: within the 16,000 characters a document may hold today
sentence = st.lists(st.sampled_from(WORDS), min_size=1, max_size=150).map(" ".join)


@settings(max_examples=60, deadline=None)
@given(st.lists(sentence, min_size=1, max_size=16), st.integers(200, 4000))
def test_segments_cover_the_revision_and_hold_every_unit_once(sentences, limit):
    doc = document(sentences)
    try:
        segments = segment_revision(doc, limit=limit)
    except ContractError as exc:
        assert str(exc).startswith("unsupported_segment:")
        return
    assert segments[0].start == 0 and segments[-1].end == len(doc.text)
    for a, b in itertools.pairwise(segments):
        assert a.end == b.start
    ids = [u for s in segments for u in s.unit_ids]
    assert sorted(ids) == sorted(u.id for u in doc.units) and len(ids) == len(set(ids))
    spans = {u.id: u for u in doc.units}
    for s in segments:
        assert len(s.embedding_input(doc).encode()) <= limit
        assert all(s.start <= spans[u].start and spans[u].end <= s.end for u in s.unit_ids)
        assert "".join(x.text(doc) for x in segments) == doc.text
    assert segment_revision(doc, limit=limit) == segments  # deterministic


def test_a_revision_that_fits_is_one_segment_with_todays_input():
    doc = document(["The budget of project P is 500.", "It was approved in 2025."])
    (only,) = segment_revision(doc)
    assert (only.segment_id, only.start, only.end) == ("s0", 0, len(doc.text))
    assert only.embedding_input(doc) == embedding_input(doc)


def test_a_long_revision_is_split_at_unit_boundaries_within_the_limit():
    doc = document(["예산 " * 700 + "끝."] * 6)  # about 6 KB per unit in UTF-8
    segments = segment_revision(doc)
    assert len(segments) == 6 and all(len(s.unit_ids) == 1 for s in segments)
    assert all(len(s.embedding_input(doc).encode()) <= LIMIT_BYTES for s in segments)


def test_a_unit_longer_than_the_limit_is_refused_not_cut():
    doc = document(["short.", "예산 " * 1500 + "끝."])
    with pytest.raises(ContractError, match="unsupported_segment:unit_too_long:u1"):
        segment_revision(doc)


def test_uncited_text_that_would_overflow_a_segment_is_refused():
    doc = document(["a.", "b."], gap=" " + "x" * 9000 + " ")
    with pytest.raises(ContractError, match="unsupported_segment:uncited_text_too_long"):
        segment_revision(doc)


def test_the_title_counts_against_the_limit_and_is_kept_in_every_segment():
    title = "T" * 400
    doc = document(["word " * 1000] * 3, title=title)
    segments = segment_revision(doc)
    assert all(s.embedding_input(doc).startswith(title + "\n") for s in segments)
    assert all(len(s.embedding_input(doc).encode()) <= LIMIT_BYTES for s in segments)


def test_the_limit_leaves_room_for_the_configured_document_prefix():
    from pathlib import Path

    from rvg.config import load_settings
    from rvg.storage.segments import PREFIX_RESERVE_BYTES, PROVIDER_LIMIT_BYTES

    settings = load_settings(Path(__file__).resolve().parents[1])
    prefix = settings.embedding_document_prefix or ""
    assert len(prefix.encode()) <= PREFIX_RESERVE_BYTES
    assert LIMIT_BYTES + len(prefix.encode()) <= PROVIDER_LIMIT_BYTES
