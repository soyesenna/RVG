"""N6-2 (parent docs/N6_2_TECHQA_PROTOCOL.md §11): one document's tied units go in document order,
not in unit id string order (u10 before u2)."""

from rvg.universal.engine import rank_units


def unit(doc, k, start, rev=1):
    return {"id": f"{doc}@{rev}#u{k}", "document_id": doc, "revision": rev, "start": start}


def test_tied_units_of_one_document_follow_the_text_not_the_id_string():
    units = [unit("d", k, 100 * k) for k in (0, 1, 10, 11, 2, 3)]
    ranked = rank_units(units, {}, {"d@1": 0})
    assert [u["id"].split("#")[1] for u in ranked] == ["u0", "u1", "u2", "u3", "u10", "u11"]


def test_seed_rank_and_segment_score_still_come_first():
    a, b = unit("a", 2, 200), unit("b", 0, 0)
    by_unit = {a["id"]: (0, -1.0)}
    assert rank_units([b, a], by_unit, {"a@1": 0, "b@1": 1}) == [a, b]
    assert rank_units([a, b], {}, {"a@1": 1, "b@1": 0}) == [b, a]
