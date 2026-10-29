import asyncio

from rvg.temporal.gate import overlap, select_evidence
from rvg.temporal.models import TemporalDocument, TemporalQuestion


def document(identifier, text, *, day, family="f", supersedes=()):
    return TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family,
            "revision": 1,
            "title": "Club",
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": f"2023-02-{day}T00:00:00Z",
            "observed_at": f"2023-02-{day}T00:00:00Z",
            "units": [{"id": "u0", "start": 0, "end": len(text)}],
            "supersedes": list(supersedes),
        }
    )


OLD = "The color commentators for the club are Erik Kramer and Tom Thayer, who joined in 2013."
NEW = "The color commentators for the club are Jim Miller and Tom Thayer, who joined in 2013."
OTHER = "Ticket prices for the 2023 season rose by four percent across all sections."


def active_ids(docs, **kwargs):
    question = TemporalQuestion(
        text="Who are the color commentators?",
        as_of="2023-03-01T00:00:00Z",
        known_at="2023-03-01T00:00:00Z",
    )
    return sorted(e["id"] for e in asyncio.run(select_evidence(docs, question, **kwargs)).active)


def test_a_quiet_revision_supersedes_the_older_document_only_when_the_rule_is_on():
    docs = [document("a-r1", OLD, day="09"), document("a-r2", NEW, day="10")]
    assert overlap(docs[0], docs[1]) > 0.9
    # Off (the policy default): no judgement, both versions stay active.
    assert active_ids(docs) == ["a-r1@1#u0", "a-r2@1#u0"]
    # On: the later, overlapping document is a revision; the older unit is superseded by code.
    assert active_ids(docs, revision_overlap=0.6) == ["a-r2@1#u0"]


def test_the_rule_never_fires_on_unrelated_text_other_families_or_older_documents():
    a, b = document("a-r1", OLD, day="09"), document("b", OTHER, day="10")
    assert overlap(a, b) < 0.6
    assert active_ids([a, b], revision_overlap=0.6) == ["a-r1@1#u0", "b@1#u0"]
    other_family = document("c", NEW, day="10", family="g")
    assert active_ids([a, other_family], revision_overlap=0.6) == ["a-r1@1#u0", "c@1#u0"]
    # Direction follows publication: the later of two overlapping same-family documents is the
    # revision, whichever id carries the newer text.
    earlier = document("d", NEW, day="08")
    assert active_ids([a, earlier], revision_overlap=0.6) == ["a-r1@1#u0"]


class Judge:
    """A pair judge that answers every relation question with one fixed verdict."""

    def __init__(self, choice, probability):
        self.choice, self.probability = choice, probability
        self.trace = []

    async def ask(self, stage, state, questions):
        answers = {}
        for key, question in questions.items():
            names = list(question["criteria"])
            spread = {n: (1 - self.probability) / (len(names) - 1) for n in names}
            spread[self.choice] = self.probability
            answers[key] = {"choice": self.choice, "probabilities": spread}
        return answers


def test_an_uncertain_judgement_with_high_overlap_is_a_revision_only_under_the_prior():
    docs = [document("a-r1", OLD, day="09"), document("a-r2", NEW, day="10")]
    unsure = Judge("UNKNOWN", 0.4)  # below the .85/.15 floor: semantic uncertainty
    off = active_ids(docs, gateway=unsure, revision_prior=0)  # an explicit 0 switches it off
    assert off == ["a-r1@1#u0", "a-r2@1#u0"]
    assert active_ids(docs, gateway=unsure, revision_prior=0.6) == ["a-r2@1#u0"]
    # The policy (U10: theta .8, document scope) resolves it as well.
    assert active_ids(docs, gateway=unsure) == ["a-r2@1#u0"]
    # A confident CONFLICTS keeps both versions active whatever the overlap: the judge's veto.
    conflict = Judge("CONFLICTS", 0.95)
    assert active_ids(docs, gateway=conflict, revision_prior=0.6) == ["a-r1@1#u0", "a-r2@1#u0"]


def test_the_document_scope_prior_retires_the_whole_older_revision():
    old_text = OLD + " " + OTHER
    new_text = NEW + " " + OTHER
    docs = [
        TemporalDocument.model_validate(
            {
                "id": identifier,
                "family": "f",
                "revision": 1,
                "title": "Club",
                "text": text,
                "source": f"fixture://{identifier}",
                "published_at": f"2023-02-{day}T00:00:00Z",
                "observed_at": f"2023-02-{day}T00:00:00Z",
                "units": [
                    {"id": "u0", "start": 0, "end": len(text.split(" " + OTHER)[0])},
                    {"id": "u1", "start": len(text) - len(OTHER), "end": len(text)},
                ],
            }
        )
        for identifier, text, day in (("a-r1", old_text, "09"), ("a-r2", new_text, "10"))
    ]
    unsure = Judge("UNKNOWN", 0.4)
    # Unit scope: only the judged units go; a per-unit UNKNOWN on both retires both here, but a
    # confident UNRELATED on the unchanged unit would keep it. Document scope retires the whole
    # older revision once any of its units is a quiet revision.
    assert active_ids(
        docs, gateway=unsure, revision_prior=0.6, revision_prior_scope="document"
    ) == [
        "a-r2@1#u0",
        "a-r2@1#u1",
    ]
