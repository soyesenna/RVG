"""0.1.0 (parent docs/PUB_1_REPORT.md): a ranking and its scores are the same bits on every
platform.

A BLAS matrix-vector product sums in the library's own order (Accelerate's also depends on the CPU
and on the block), so scores ranked straight from BLAS had other last bits on another machine, and
rows a rounding apart could swap. Canonical scores multiply and sum in numpy's own pairwise order;
a scan keeps BLAS for speed and rescores canonically every row whose scan score is within twice
`scan_bound` of the k-th. These tests pin the summation order to a pure-Python rendering of numpy's
pairwise sum, and stand in for another platform's BLAS by moving every scan score within the
bound: the ranking and its scores must not move by one bit."""

import math
from itertools import pairwise as successive
from types import SimpleNamespace

import numpy as np
import pytest

from rvg.storage import incremental, local_exact
from rvg.storage.incremental import rank_parts
from rvg.storage.local_exact import canonical_scores, rank_rows, scan_bound, unit


def bits(a):
    return np.asarray(a, dtype=np.float64).view(np.uint64)


def pairwise(values):
    """numpy's pairwise sum (`pairwise_sum`, numpy/_core/src/umath/loops_utils.h.src) in Python
    floats: below 8 terms one by one, up to 128 in eight running sums, else split in two halves
    cut at a multiple of 8."""
    n = len(values)
    if n < 8:
        total = 0.0
        for v in values:
            total += v
        return total
    if n <= 128:
        r = list(values[:8])
        i = 8
        while i < n - n % 8:
            for j in range(8):
                r[j] += values[i + j]
            i += 8
        total = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
        for v in values[i:]:
            total += v
        return total
    half = n // 2
    half -= half % 8
    return pairwise(values[:half]) + pairwise(values[half:])


def moved(monkeypatch, module, seed, scale=0.9):
    """Another platform's BLAS: every scan score moved by up to `scale` x `scan_bound`."""
    rng = np.random.default_rng(seed)
    real = module.scan_scores

    def scan(block, q):
        scores = real(block, q)
        return scores + rng.uniform(-scale, scale, scores.shape) * scan_bound(q.size)

    monkeypatch.setattr(module, "scan_scores", scan)


def near_ties(n, dims, seed):
    """Unit rows with exact duplicates and rows one rounding away from another row."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, dims))
    m[n // 2 : n // 2 + 40] = m[7]  # duplicates: exact ties in catalog order
    m[n // 2 + 40 : n // 2 + 80] = m[11] * (1 + 1e-15 * rng.standard_normal((40, 1)))
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def brute(matrix_row, matrix, mask, query, top_k):
    """The definition: canonical scores of every eligible row, stable descending order."""
    ready = np.flatnonzero(mask & (matrix_row >= 0))
    scores = canonical_scores(matrix[matrix_row[ready]], unit(query))
    order = np.argsort(-scores, kind="stable")[:top_k]
    return ready[order], scores[order]


@pytest.mark.parametrize("dims", [1, 3, 8, 9, 64, 129, 300, 1536])
def test_canonical_scores_are_numpy_pairwise_sums_of_rounded_products(dims):
    rng = np.random.default_rng(dims)
    m = rng.standard_normal((25, dims)) * rng.uniform(1e-3, 1e3, (25, 1))
    q = rng.standard_normal(dims)
    want = [pairwise([float(x) * float(y) for x, y in zip(row, q, strict=True)]) for row in m]
    assert np.array_equal(bits(canonical_scores(m, q)), bits(want))
    norm = math.sqrt(pairwise([float(x) * float(x) for x in q]))
    assert np.array_equal(bits(unit(q)), bits([float(x) / norm for x in q]))


def test_a_rows_canonical_score_depends_on_no_other_row():
    m = np.random.default_rng(1).standard_normal((1300, 96))
    q = unit(np.random.default_rng(2).standard_normal(96))
    full = canonical_scores(m, q)
    for rows in (np.arange(5, 9), np.arange(600, 1300), np.array([1299, 3, 3, 700])):
        assert np.array_equal(bits(canonical_scores(m[rows], q)), bits(full[rows]))
    wide = np.random.default_rng(3).standard_normal((1300, 192))
    wide[:, :96] = m
    assert np.array_equal(bits(canonical_scores(wide[:, :96], q)), bits(full))  # a strided view
    # a vector stored as a matrix row and the same vector as a query normalise to the same bits
    assert np.array_equal(
        bits(unit(m[4])), bits(m[4:5] / np.linalg.norm(m[4:5], axis=1)[:, None])[0]
    )


@pytest.mark.parametrize("dims", [64, 1536])
def test_rank_rows_does_not_move_when_the_scan_moves_within_its_bound(monkeypatch, dims):
    matrix = near_ties(3000, dims, dims)
    matrix_row = np.random.default_rng(5).permutation(3000)  # catalog order is not row order
    rng = np.random.default_rng(6)
    cases = []
    for i in range(30):
        query = matrix[[7, 11, 900][i % 3]] + 0.05 * rng.standard_normal(dims)
        mask = rng.random(3000) < (0.9 if i % 2 else 0.2)  # full scan / gathered rows
        cases.append((mask, query, [1, 5, 40, 120][i % 4]))
    before = [rank_rows(matrix_row, matrix, m, q, k) for m, q, k in cases]
    for (m, q, k), (rows, scores) in zip(cases, before, strict=True):
        want = brute(matrix_row, matrix, m, q, k)
        assert np.array_equal(rows, want[0]) and np.array_equal(bits(scores), bits(want[1]))
    for seed in range(3):
        monkeypatch.undo()  # each seed moves the BLAS scores once
        moved(monkeypatch, local_exact, seed)
        for (m, q, k), (rows, scores) in zip(cases, before, strict=True):
            got = rank_rows(matrix_row, matrix, m, q, k)
            assert np.array_equal(got[0], rows) and np.array_equal(bits(got[1]), bits(scores))


def test_rank_parts_does_not_move_when_the_scan_moves_within_its_bound(monkeypatch):
    matrix = near_ties(2500, 64, 8)
    cuts = (0, 700, 701, 1900, 2500)  # a part of one row too
    parts = [(lo, SimpleNamespace(matrix=matrix[lo:hi])) for lo, hi in successive(cuts)]
    matrix_row = np.random.default_rng(9).permutation(2500)
    rng = np.random.default_rng(10)
    cases = [
        (rng.random(2500) < (0.9 if i % 2 else 0.1), matrix[7] + 0.05 * rng.standard_normal(64))
        for i in range(12)
    ]
    before = [rank_parts(matrix_row, parts, m, q, 60) for m, q in cases]
    for (m, q), (rows, scores) in zip(cases, before, strict=True):
        want = brute(matrix_row, matrix, m, q, 60)
        assert np.array_equal(rows, want[0]) and np.array_equal(bits(scores), bits(want[1]))
    moved(monkeypatch, incremental, 11)
    for (m, q), (rows, scores) in zip(cases, before, strict=True):
        got = rank_parts(matrix_row, parts, m, q, 60)
        assert np.array_equal(got[0], rows) and np.array_equal(bits(got[1]), bits(scores))


def test_the_bound_covers_float32_scans_and_grows_with_the_dimensions():
    assert scan_bound(1536) < 1e-11 < scan_bound(1536, np.float32) < 1e-3
    assert scan_bound(64) < scan_bound(1536)
    rng = np.random.default_rng(12)
    m = rng.standard_normal((4000, 1536))
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    q = unit(rng.standard_normal(1536))
    exact = canonical_scores(m, q)
    assert np.abs(m @ q - exact).max() <= scan_bound(1536)
    low = (m.astype(np.float32) @ q.astype(np.float32)).astype(np.float64)
    assert np.abs(low - exact).max() <= scan_bound(1536, np.float32)
