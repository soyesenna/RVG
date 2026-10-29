"""RVI-02: validity is decided over the tenant's authoritative state, content by the ACL. A
revision that replaces a retrieved document by a `supersedes` declaration (or a higher revision of
it) takes the replaced units out of force even when the principal cannot see it, it was revoked,
the user's selection left it out or retrieval did not return it — by code rules on its metadata,
never by showing its text. The public view never names it.

The fake provider answers with the first evidence unit that states a number; the gate's decisions
are read from the engine's result, leaks from the fields a caller publishes (`PUBLISHED`, the KB-FF
response fields). This build has no public view of its own (RVI-01 is not in it): `answer` returns
the engine's result, whose audit is not a published field."""

from __future__ import annotations

import asyncio

import pytest
from session_fixtures import budget, fake_session, validate

import rvg

T = "fixture"
STAFF = rvg.TrustedPrincipal("staff", ("employees",))
FINANCE = rvg.TrustedPrincipal("finance", ("employees", "finance-team"))
AS_OF = "2026-09-20T00:00:00Z"


def doc(
    identifier,
    sentences,
    *,
    family="cap",
    published="2026-07-20T00:00:00Z",
    observed=None,
    effective_from=None,
    supersedes=(),
    authority="primary",
    revision=1,
):
    text, units, offset = " ".join(sentences), [], 0
    for i, part in enumerate(sentences):
        units.append({"id": f"u{i}", "start": offset, "end": offset + len(part)})
        offset += len(part) + 1
    return rvg.TemporalDocument.model_validate(
        {
            "id": identifier,
            "family": family,
            "revision": revision,
            "title": identifier.upper(),
            "text": text,
            "source": f"fixture://{identifier}",
            "published_at": published,
            "observed_at": observed or published,
            "effective_from": effective_from,
            "authority": authority,
            "units": units,
            "supersedes": [{"document_id": d, "unit_ids": list(u)} for d, u in supersedes],
        }
    )


A = doc("a-cap", ["The lodging cap is 100 USD per night.", "Claims are filed within 30 days."])


def amendment(**kw):
    base = {"published": "2026-09-10T00:00:00Z", "supersedes": (("a-cap", ()),)}
    return doc("b-amend", ["The lodging cap is 120 USD per night."], **(base | kw))


def ctx(principal=None, **kw):
    return rvg.RequestContext(tenant=T, principal=principal, call=budget(), **kw)


def ask(text="What is the lodging cap?", as_of=AS_OF, known_at=None):
    return {"text": text, "as_of": as_of, "known_at": known_at or as_of}


async def world(session, *items):
    await session.ingest(list(items), ctx())


def statuses(native):
    """unit id -> gate status, over every validity audit of the answer."""
    out = {}
    for audit in native.get("audit") or []:
        out.update(audit["gate"]["statuses"])
    return out


def run(setup, principal=STAFF, question=None, selected=None):
    async def go():
        async with fake_session() as (session, _, store):
            await setup(session, store)
            c = {"principal": principal}
            if selected is not None:
                c["selected_document_ids"] = tuple(selected)
            envelope = await session.answer(question or ask(), ctx(**c))
            validate(envelope, "answer_envelope")
            return published(envelope), envelope["result"]

    return asyncio.run(go())


PUBLISHED = (
    "status",
    "reason",
    "text",
    "short_answer",
    "sentences",
    "evidence_ids",
    "missing",
    "answer_kind",
    "release",
    "snapshot",
)


def published(envelope):
    """What a caller publishes from `answer` (as the KB-FF adapter does): the result's public
    fields, `short_answer` only for FULL/PARTIAL — not the audit, the trace or a withheld draft."""
    result = envelope["result"]
    out = {k: result.get(k) for k in PUBLISHED}
    if result.get("status") not in ("FULL", "PARTIAL"):
        out["short_answer"] = None
    return {"result": out}  # not the random request id: the leak checks look for short numbers


def restricted(document, allow=("finance",)):
    return {"document": document.model_dump(mode="json"), "access": {"allow": list(allow)}}


def no_leak(public, *needles):
    blob = repr(public)
    return not any(n in blob for n in needles)


def test_v01_without_an_amendment_the_visible_value_is_answered():
    public, _ = run(lambda s, _: world(s, A))
    assert public["result"]["status"] == "FULL" and "100" in public["result"]["text"]


def test_v02_a_hidden_amendment_takes_the_old_value_out_of_force():
    async def setup(session, _):
        await world(session, A, restricted(amendment()))

    public, native = run(setup)
    assert statuses(native)["a-cap@1#u0"] == "SUPERSEDED"
    assert public["result"]["status"] == "ABSTAIN" and "100" not in repr(public)
    assert no_leak(public, "b-amend", "120")
    # the authorised principal gets the amended value
    finance, _ = run(setup, principal=FINANCE)
    assert finance["result"]["status"] == "FULL" and "120" in finance["result"]["text"]


def test_v03_once_the_amendment_is_visible_everyone_gets_the_amended_value():
    public, _ = run(lambda s, _: world(s, A, amendment()))
    assert public["result"]["status"] == "FULL" and "120" in public["result"]["text"]


def test_v04_a_hidden_document_without_a_declaration_changes_nothing():
    other = doc(
        "c-other", ["The lodging cap is 999 USD per night."], published="2026-09-12T00:00:00Z"
    )

    async def setup(session, _):
        await world(session, A, restricted(other))

    public, native = run(setup)
    assert statuses(native)["a-cap@1#u0"] == "ELIGIBLE"
    assert public["result"]["status"] == "FULL" and "100" in public["result"]["text"]
    assert no_leak(public, "c-other", "999")


def test_v05_a_partial_amendment_replaces_only_its_unit():
    async def setup(session, _):
        await world(session, A, restricted(amendment(supersedes=(("a-cap", ("u0",)),))))

    _, native = run(setup)
    got = statuses(native)
    assert got["a-cap@1#u0"] == "SUPERSEDED" and got["a-cap@1#u1"] == "ELIGIBLE"


def test_v06_a_future_amendment_applies_from_its_start():
    future = amendment(effective_from="2026-10-01T00:00:00Z")

    async def setup(session, _):
        await world(session, A, restricted(future))

    _, before = run(setup)
    _, after = run(setup, question=ask(as_of="2026-10-02T00:00:00Z"))
    assert statuses(before)["a-cap@1#u0"] == "ELIGIBLE"
    assert statuses(after)["a-cap@1#u0"] == "SUPERSEDED"


def test_v07_a_late_arrival_applies_only_after_it_was_known():
    late = amendment(observed="2026-09-15T00:00:00Z")

    async def setup(session, _):
        await world(session, A, restricted(late))

    _, then = run(setup, question=ask(known_at="2026-09-12T00:00:00Z"))
    _, now = run(setup)
    assert statuses(then)["a-cap@1#u0"] == "ELIGIBLE"
    assert statuses(now)["a-cap@1#u0"] == "SUPERSEDED"


def test_v08_revoking_the_amendment_does_not_resurrect_the_old_value():
    async def setup(session, store):
        await world(session, A, amendment())
        await session.apply_event({"type": "revoke", "document_id": "b-amend"}, ctx())

    public, native = run(setup)
    assert statuses(native)["a-cap@1#u0"] == "SUPERSEDED"
    assert public["result"]["status"] == "ABSTAIN" and no_leak(public, "b-amend", "120")


@pytest.mark.parametrize(
    "kw",
    [
        {"published": "2026-07-01T00:00:00Z"},  # declared by an earlier document
        {"authority": "secondary"},  # a secondary source cannot replace a primary one
    ],
    ids=["order", "authority"],
)
def test_v09_v19_an_invalid_declaration_leaves_nothing_in_force_rather_than_a_guess(kw):
    async def setup(session, _):
        await world(session, A, restricted(amendment(**kw)))

    public, _ = run(setup)
    assert public["result"]["status"] == "ABSTAIN" and no_leak(public, "b-amend", "120")
    assert "100" not in public["result"]["text"]


def test_v11_an_amendment_arriving_during_the_answer_is_not_skipped():
    async def go():
        async with fake_session() as (session, fake, store):
            await world(session, A)
            original = fake.writer

            def writer(payload):
                store.upsert(T, amendment())  # committed while the draft is being written
                fake.writer = original
                return original(payload)

            fake.writer = writer
            return await session.answer(ask(), ctx(principal=STAFF))

    public = asyncio.run(go())
    # U24 (0.1.0.dev27): the default release check sees the amendment of a document the draft
    # read and regenerates once on the amended state; `current` (the default before) withheld the
    # draft as source_changed. Either way the draft written before the amendment is not released.
    result = public["result"]
    assert result["status"] == "FULL" and result["short_answer"] == "120"
    assert result["evidence_ids"] == ["b-amend@1#u0"]  # the regeneration read the amendment
    assert result["release"]["mode"] == "scoped_dependency"


def test_v12_v21_a_declaration_retrieval_or_the_users_selection_left_out_still_applies():
    async def setup(session, _):
        await world(session, A, amendment())

    public, native = run(setup, selected=["a-cap"])
    assert statuses(native)["a-cap@1#u0"] == "SUPERSEDED"
    assert public["result"]["status"] == "ABSTAIN"


def test_v20_paired_worlds_leak_no_content_or_identifier_and_the_status_difference_is_recorded():
    w0, _ = run(lambda s, _: world(s, A))

    async def setup(session, _):
        await world(session, A, restricted(amendment()))

    w1, _ = run(setup)
    assert w0["result"]["status"] == "FULL" and w1["result"]["status"] == "ABSTAIN"
    assert no_leak(w1, "b-amend", "120", "B-AMEND")
    # the existence signal that remains: status, reason and the tenant-wide knowledge number
    assert w1["result"]["reason"] == "no_active_evidence"
    assert w1["result"]["snapshot"] == w0["result"]["snapshot"] + 1


def test_the_plane_reads_revisions_written_before_the_index_existed(tmp_path):
    path = tmp_path / "old.sqlite3"
    store = rvg.TemporalStore(path)
    store.upsert(T, A)
    store.upsert(T, amendment())
    store.conn.execute("DELETE FROM temporal_supersessions")  # as a version before RVI-02 left it
    store.conn.commit()
    store.close()
    reopened = rvg.TemporalStore(path)
    try:
        found = reopened.supersession_sources(T, {"a-cap"})
        assert [d.key for d in found] == ["a-cap@1", "b-amend@1"]
    finally:
        reopened.close()


def _sqlite():
    return rvg.TemporalStore(":memory:")


def _postgres():
    import os
    import uuid

    from rvg.storage.postgres import PostgresState

    return PostgresState(
        os.environ["DEEPJEV_TEST_POSTGRES_DSN"], schema="rvi02_" + uuid.uuid4().hex[:10]
    )


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(_sqlite, id="sqlite"),
        pytest.param(
            _postgres,
            id="postgres",
            marks=pytest.mark.skipif(
                not __import__("os").environ.get("DEEPJEV_TEST_POSTGRES_DSN"),
                reason="DEEPJEV_TEST_POSTGRES_DSN not set",
            ),
        ),
    ],
)
def test_the_state_lists_declarations_whatever_their_acl_or_revocation(make):
    state = make()
    try:
        state.upsert(T, A)
        state.replace_access(T, "b-amend", {"finance": "allow"})
        state.upsert(T, amendment())
        state.revoke(T, "b-amend")
        assert [d.key for d in state.supersession_sources(T, {"a-cap"})] == ["a-cap@1", "b-amend@1"]
        assert [d.key for d in state.supersession_sources(T, {"a-cap"}, at=1)] == ["a-cap@1"]
        assert state.supersession_sources("other-tenant", {"a-cap"}) == []
    finally:
        state.close()


# RVI-02b: an explicit declaration applies whatever the two documents' families are (KB-FF
# R-REV-2); its order and authority still decide whether it is valid.


def other_family(**kw):
    return amendment(family="lodging-2026", **kw)


def test_02b_1_a_hidden_declaration_of_another_family_applies():
    async def setup(session, _):
        await world(session, A, restricted(other_family()))

    public, native = run(setup)
    assert statuses(native)["a-cap@1#u0"] == "SUPERSEDED"
    assert public["result"]["status"] == "ABSTAIN" and "100" not in repr(public)
    assert no_leak(public, "b-amend", "120", "B-AMEND", "lodging-2026")
    finance, _ = run(setup, principal=FINANCE)
    assert finance["result"]["status"] == "FULL" and "120" in finance["result"]["text"]
    assert "100" not in finance["result"]["text"]


def test_02b_2_a_visible_declaration_of_another_family_applies():
    public, native = run(lambda s, _: world(s, A, other_family()))
    assert statuses(native)["a-cap@1#u0"] == "SUPERSEDED"
    assert public["result"]["status"] == "FULL" and "120" in public["result"]["text"]
    assert "100" not in public["result"]["text"]


@pytest.mark.parametrize("hidden", [True, False], ids=["hidden", "visible"])
@pytest.mark.parametrize(
    "kw",
    [{"published": "2026-07-01T00:00:00Z"}, {"authority": "secondary"}],
    ids=["order", "authority"],
)
def test_02b_3_4_an_invalid_declaration_of_another_family_still_leaves_nothing_in_force(kw, hidden):
    async def setup(session, _):
        document = other_family(**kw)
        await world(session, A, restricted(document) if hidden else document)

    public, native = run(setup)
    assert statuses(native)["a-cap@1#u0"] == "ELIGIBLE"  # not applied ...
    assert public["result"]["status"] == "ABSTAIN"  # ... and not guessed either
    assert "100" not in public["result"]["text"]
    if hidden:
        assert no_leak(public, "b-amend", "120", "B-AMEND")


def test_02b_5_a_declaration_of_a_unit_the_target_lacks_is_unresolved():
    async def setup(session, _):
        await world(session, A, restricted(other_family(supersedes=(("a-cap", ("u9",)),))))

    public, native = run(setup)
    assert statuses(native)["a-cap@1#u0"] == "ELIGIBLE"
    assert public["result"]["status"] == "ABSTAIN" and "100" not in public["result"]["text"]
    assert no_leak(public, "b-amend", "120")


def test_02b_6_a_declaration_in_another_tenant_changes_nothing_here():
    async def setup(session, _):
        await world(session, A)
        elsewhere = rvg.RequestContext(tenant="elsewhere", call=budget())
        await session.ingest([other_family()], elsewhere)

    public, native = run(setup)
    assert statuses(native)["a-cap@1#u0"] == "ELIGIBLE"
    assert public["result"]["status"] == "FULL" and "100" in public["result"]["text"]


def test_02b_7_revoking_another_familys_declaration_does_not_resurrect_the_old_value():
    async def setup(session, _):
        await world(session, A, other_family())
        await session.apply_event({"type": "revoke", "document_id": "b-amend"}, ctx())

    public, native = run(setup)
    assert statuses(native)["a-cap@1#u0"] == "SUPERSEDED"
    assert public["result"]["status"] == "ABSTAIN" and no_leak(public, "b-amend", "120")
