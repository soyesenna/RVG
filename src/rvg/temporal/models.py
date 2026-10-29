from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from ..contracts import Contract


class SourceSpan(Contract):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[\w-]+$")
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class Replacement(Contract):
    document_id: str = Field(min_length=1)
    unit_ids: tuple[str, ...] = ()  # Empty means the entire target document.


class TemporalDocument(Contract):
    id: str = Field(min_length=1, max_length=160, pattern=r"^[\w-]+$")
    family: str = Field(min_length=1, max_length=160)
    revision: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=500)
    # K8-1: a long revision is segmented for retrieval (storage/segments.py), never cut.
    text: str = Field(min_length=1, max_length=400_000)
    published_at: AwareDatetime
    observed_at: AwareDatetime
    effective_from: AwareDatetime | None = None
    effective_until: AwareDatetime | None = None
    authority: Literal["primary", "secondary"] = "primary"
    source: str = Field(min_length=1, max_length=1000)
    units: tuple[SourceSpan, ...] = Field(min_length=1, max_length=4096)
    supersedes: tuple[Replacement, ...] = ()

    @model_validator(mode="after")
    def integrity(self):
        if self.observed_at < self.published_at:
            raise ValueError("observation_before_publication")
        if self.effective_until and self.effective_until <= self.starts:
            raise ValueError("invalid_effective_interval")
        if len({u.id for u in self.units}) != len(self.units):
            raise ValueError("duplicate_source_unit")
        for unit in self.units:
            if (
                not unit.start < unit.end <= len(self.text)
                or not self.text[unit.start : unit.end].strip()
            ):
                raise ValueError("invalid_source_span")
        if any(r.document_id == self.id for r in self.supersedes):
            raise ValueError("self_replacement_use_revision")
        return self

    @property
    def starts(self):
        return self.effective_from or self.published_at

    @property
    def key(self):
        return f"{self.id}@{self.revision}"

    def evidence(self):
        return [
            {
                "id": f"{self.key}#{u.id}",
                "document_id": self.id,
                "revision": self.revision,
                "family": self.family,
                "title": self.title,
                "text": self.text[u.start : u.end],
                "source": self.source,
                "source_sha256": hashlib.sha256(self.text.encode()).hexdigest(),
                "start": u.start,
                "end": u.end,
                "published_at": self.published_at.isoformat(),
                "effective_from": self.starts.isoformat(),
                "effective_until": self.effective_until.isoformat()
                if self.effective_until
                else None,
                "authority": self.authority,
            }
            for u in self.units
        ]


class TemporalQuestion(Contract):
    text: str = Field(min_length=1, max_length=2000)
    as_of: AwareDatetime
    known_at: AwareDatetime
    # BCP-47 tag of the language to answer in; it never selects a policy.
    locale: str = Field(default="en-US", min_length=2, max_length=35, pattern=r"^[A-Za-z0-9-]+$")


class AnswerSentence(Contract):
    text: str = Field(min_length=1, max_length=1600)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=12)


class AnswerDraft(Contract):
    answerable: bool
    short_answer: str = Field(max_length=1000)
    sentences: tuple[AnswerSentence, ...] = Field(max_length=8)


class SentenceClaims(Contract):
    sentence_index: int = Field(ge=0)
    claims: tuple[str, ...] = Field(min_length=1, max_length=6)


class ExtractedClaims(Contract):
    sentences: tuple[SentenceClaims, ...] = Field(min_length=1, max_length=8)
    short_answer_claims: tuple[str, ...] = Field(min_length=1, max_length=8)
