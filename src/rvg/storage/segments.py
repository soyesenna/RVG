"""K8-1: retrieval segments of a revision (docs/K8_1_PROTOCOL.md).

R1 `units-greedy/1`: consecutive source units, in text order, are packed into one segment while
the segment's embedding input (title + newline + segment text, `title-newline-text/1`) stays
within the byte limit the embedding provider accepts. A segment's text runs from where the
previous one ended to the end of its last unit — the first from 0, the last to the end of the
text — so the segments cover the revision without gap or overlap and every unit lies in exactly
one segment. Units are the citation units and are never split here: a unit whose own input
exceeds the limit blocks the revision (`unsupported_segment`), and the source adapter must give
finer units (sentences, table rows, turns). Nothing is cut silently.

A revision whose whole input fits is one segment `s0` spanning the whole text: its input is the
revision's `embedding_input`, so today's vectors are its vectors (no re-embedding), and its
catalog row is the revision itself with the K8-0 whole-revision identity (`s0`,
`whole-revision/1`) — keys, index points and results as before.

A longer revision is one catalog row per segment, each a `SegmentView`: the revision seen
through that segment's units. Views of one revision that a search selects together merge."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from ..errors import ContractError

SEGMENTATION = "units-greedy/1"
# The embedding provider refuses an input over 8,192 bytes after it prepends the space's document
# prefix (`title: none | text: ` today, 20 bytes); segments leave a fixed reserve for any prefix,
# so the rule does not depend on a deployment. A longer prefix fails loudly at indexing.
PROVIDER_LIMIT_BYTES, PREFIX_RESERVE_BYTES = 8192, 256
LIMIT_BYTES = PROVIDER_LIMIT_BYTES - PREFIX_RESERVE_BYTES


@dataclass(frozen=True)
class Segment:
    segment_id: str
    ordinal: int
    start: int
    end: int
    unit_ids: tuple[str, ...]

    def text(self, document):
        return document.text[self.start : self.end]

    def embedding_input(self, document):
        return document.title + "\n" + self.text(document)


def _size(document, start, end):
    return len((document.title + "\n" + document.text[start:end]).encode("utf-8"))


def segment_revision(document, *, limit=LIMIT_BYTES):
    """The segments of one revision (see the module docstring)."""
    if _size(document, 0, len(document.text)) <= limit:
        return (Segment("s0", 0, 0, len(document.text), tuple(u.id for u in document.units)),)
    units = sorted(document.units, key=lambda u: (u.start, u.end))
    for a, b in itertools.pairwise(units):
        if b.start < a.end:
            raise ContractError(f"unsupported_segment:overlapping_units:{a.id}:{b.id}")
    segments, start, group = [], 0, []
    for unit in units:
        if _size(document, unit.start, unit.end) > limit:
            raise ContractError(f"unsupported_segment:unit_too_long:{unit.id}")
        if group and _size(document, start, unit.end) > limit:
            end = group[-1].end
            segments.append((start, end, group))
            start, group = end, []
        group.append(unit)
    segments.append((start, len(document.text), group))
    out = []
    for n, (a, b, members) in enumerate(segments):
        if _size(document, a, b) > limit:
            # The trailing text after the last unit (or the gap before the first) made the
            # segment too long: the text outside any unit is not citable, so it is not kept in
            # a segment that would exceed the limit.
            raise ContractError(f"unsupported_segment:uncited_text_too_long:s{n}")
        out.append(Segment(f"s{n}", n, a, b, tuple(u.id for u in members)))
    return tuple(out)


class SegmentView:
    """The part of a revision a catalog row stands for, handed to the engine like a document.
    Its units are the segments' units (evidence ids stay `{id}@{revision}#{unit}`); `text` stays
    the whole revision (unit offsets and the revision-level U10 overlap are unchanged);
    `search_text` is what ranks it lexically; `model_dump` gives the revision judge only these
    segments' text (K8 R3: never the whole long revision)."""

    def __init__(self, document, segments):
        self.document = document
        self.segments = tuple(sorted(set(segments), key=lambda s: s.ordinal))
        wanted = {u for s in self.segments for u in s.unit_ids}
        self.units = tuple(u for u in document.units if u.id in wanted)
        self.search_text = "\n".join(s.text(document) for s in self.segments)
        self.row_key = f"{document.key}~" + "+".join(s.segment_id for s in self.segments)

    def __getattr__(self, name):
        return getattr(self.document, name)

    @property
    def key(self):
        return self.document.key

    @property
    def place(self):
        """The relative text position the segments span, in [0, 1]."""
        n = max(1, len(self.document.text))
        return self.segments[0].start / n, self.segments[-1].end / n

    def evidence(self):
        wanted = {u.id for u in self.units}
        return [e for e in self.document.evidence() if e["id"].rsplit("#", 1)[1] in wanted]

    def model_dump(self, **kw):
        body = self.document.model_dump(**kw)
        body["text"] = self.search_text
        body["units"] = [{"id": u.id} for u in self.units]
        body["segments"] = [s.segment_id for s in self.segments]
        return body

    def merge(self, other):
        if other.document.key != self.document.key:
            raise ValueError("segment_views_of_different_revisions")
        return SegmentView(self.document, (*self.segments, *other.segments))


def place(row):
    """A catalog row's relative position: a whole revision spans [0, 1]."""
    return row.place if isinstance(row, SegmentView) else (0.0, 1.0)


def catalog_entries(document, *, limit=LIMIT_BYTES):
    """The catalog rows of one revision: `(row document, row key, segment id, segmentation,
    embedding input)` (see the module docstring)."""
    from .identity import SEGMENTATION as WHOLE
    from .identity import embedding_input

    segments = segment_revision(document, limit=limit)
    if len(segments) == 1:
        return [(document, document.key, "s0", WHOLE, embedding_input(document))]
    out = []
    for s in segments:
        view = SegmentView(document, [s])
        out.append((view, view.row_key, s.segment_id, SEGMENTATION, s.embedding_input(document)))
    return out


def aligned(rows, seed):
    """Of one other revision's rows, the one in the same place as `seed` (K8 R2): the largest
    overlap of relative text position, ties to the earlier segment."""
    a, b = place(seed)

    def score(row):
        c, d = place(row)
        return (max(0.0, min(b, d) - max(a, c)), -c)

    return max(rows, key=score)


def merged(rows):
    """Selected rows as the engine's documents, one per revision, in first-seen order."""
    out = {}
    for row in rows:
        have = out.get(row.key)
        if have is None:
            out[row.key] = row
        elif isinstance(have, SegmentView) and isinstance(row, SegmentView):
            out[row.key] = have.merge(row)
    return list(out.values())


def unit_ranks(documents, seed_ids, text):
    """Evidence id -> sort key for the units of segment views: the rank of the seed row its
    segment was found by, then the unit's BM25 score for the search `text` among that segment's
    units (higher first). A segment has no per-unit score of its own, and without this a
    policy's `write_top` would take its first units in id order. Whole revisions add nothing
    here (they keep their revision rank and id order, as before)."""
    from ..retrieval.search import BM25

    order = {key: rank for rank, key in enumerate(seed_ids)}
    out = {}
    for d in documents:
        units = {u.id: u for u in getattr(d, "units", ())}
        for s in getattr(d, "segments", ()):
            rank = order.get(f"{d.key}~{s.segment_id}", len(order))
            scored = dict(
                BM25(
                    [
                        {
                            "id": u,
                            "title": "",
                            "text": d.document.text[units[u].start : units[u].end],
                        }
                        for u in s.unit_ids
                    ]
                ).search(text, len(s.unit_ids))
            )
            out.update({f"{d.key}#{u}": (rank, -scored.get(u, 0.0)) for u in s.unit_ids})
    return out
