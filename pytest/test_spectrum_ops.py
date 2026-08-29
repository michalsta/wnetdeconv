"""
Equivalence tests for the C++-delegated Spectrum operations
(sort_signals / merge_signals / __add__) against a verbatim copy of the
historical pure-Python implementations kept below (test-only).

Covers random spectra in several dimensions, duplicate positions, empty
spectra, label handling, and subclass identity; ends with a timing sanity
print (not an assert) on a ~50k-point synthetic.
"""

import time

import numpy as np
import pytest

from wnetdeconv.spectrum import Spectrum, Spectrum_1D


# ---------------------------------------------------------------------------
# Legacy reference implementations (verbatim semantics of the old Python code)
# ---------------------------------------------------------------------------


def _legacy_sort_signals(sp):
    order = np.lexsort(
        tuple(sp.positions[i, :] for i in range(sp.positions.shape[0] - 1, -1, -1))
    )
    return Spectrum(
        positions=sp.positions[:, order],
        intensities=sp.intensities[order],
        label=sp.label,
    )


def _legacy_merge_signals(sp):
    # Assumes positions are sorted (legacy contract).
    if sp.positions.shape[1] == 0:
        return sp.copy()
    cpos = sp.positions[:, 0]
    csig = 0.0
    merged_pos = []
    merged_sig = []
    for pos, sig in zip(sp.positions.T, sp.intensities):
        if not np.all(pos == cpos):
            merged_pos.append(cpos)
            merged_sig.append(csig)
            cpos = pos
            csig = 0.0
        csig += sig
    merged_pos.append(cpos)
    merged_sig.append(csig)
    return Spectrum(
        positions=np.array(merged_pos).T,
        intensities=np.array(merged_sig),
        label=sp.label,
    )


def _legacy_add(a, b):
    res = Spectrum(
        positions=np.hstack((a.positions, b.positions)),
        intensities=np.hstack((a.intensities, b.intensities)),
        label=a._combined_label(a.label, b.label),
    )
    return _legacy_merge_signals(_legacy_sort_signals(res))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _random_spectrum(rng, dim, n, dup=True, integer=False, label=None):
    if dup and n > 0:
        # Draw positions from a small grid so duplicates are common.
        pos = rng.integers(0, max(2, n // 3), size=(dim, n)).astype(float) * 0.5
    else:
        pos = rng.uniform(0, 100, size=(dim, n))
    if integer:
        ints = rng.integers(1, 50, size=n).astype(float)
    else:
        ints = rng.uniform(0.01, 5.0, size=n)
    return Spectrum(pos, ints, label=label)


def _group_canonical(sp):
    """(positions, per-group-sorted intensities) sorted lexicographically.

    Tie order among peaks sharing an identical position is implementation-
    defined (np.lexsort is stable, std::sort is not), so intensities are
    compared as sorted multisets within each equal-position group.
    """
    pos = np.asarray(sp.positions)
    ints = np.asarray(sp.intensities)
    order = np.lexsort(tuple(pos[i, :] for i in range(pos.shape[0] - 1, -1, -1)))
    pos, ints = pos[:, order], ints[order]
    if pos.shape[1] == 0:
        return pos, ints
    boundary = np.any(np.diff(pos, axis=1) != 0, axis=0)
    starts = np.concatenate(([0], np.nonzero(boundary)[0] + 1, [pos.shape[1]]))
    canon = ints.copy()
    for s, e in zip(starts[:-1], starts[1:]):
        canon[s:e] = np.sort(canon[s:e])
    return pos, canon


def _assert_spectra_equal(a, b, rtol=1e-9):
    assert type(a) is Spectrum and type(b) is Spectrum
    assert a.label == b.label
    pa, ia = np.asarray(a.positions), np.asarray(a.intensities)
    pb, ib = np.asarray(b.positions), np.asarray(b.intensities)
    assert pa.shape == pb.shape
    np.testing.assert_array_equal(pa, pb)
    np.testing.assert_allclose(ia, ib, rtol=rtol, atol=1e-12)


PARAMS = [(dim, n, dup, integer)
          for dim in (1, 2, 3)
          for n in (1, 7, 40)
          for dup in (True, False)
          for integer in (True, False)]


# ---------------------------------------------------------------------------
# sort_signals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim,n,dup,integer", PARAMS)
def test_sort_signals_matches_legacy(dim, n, dup, integer):
    rng = np.random.default_rng(dim * 1000 + n * 10 + dup * 2 + integer)
    sp = _random_spectrum(rng, dim, n, dup=dup, integer=integer, label="x")
    new = sp.sort_signals()
    old = _legacy_sort_signals(sp)
    assert type(new) is Spectrum
    assert new.label == old.label == "x"
    # Positions must agree exactly; intensity tie-order within an identical
    # position is implementation-defined, so compare group-canonical forms.
    np.testing.assert_array_equal(
        np.asarray(new.positions), np.asarray(old.positions)
    )
    _, ci_new = _group_canonical(new)
    _, ci_old = _group_canonical(old)
    np.testing.assert_allclose(ci_new, ci_old, rtol=1e-12)


def test_sort_signals_no_merge():
    # Duplicate positions must survive sorting un-merged.
    sp = Spectrum_1D([2.0, 1.0, 2.0], [1.0, 2.0, 3.0], label="d")
    out = sp.sort_signals()
    assert np.asarray(out.positions).shape[1] == 3


# ---------------------------------------------------------------------------
# merge_signals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim,n,dup,integer", PARAMS)
def test_merge_signals_matches_legacy(dim, n, dup, integer):
    rng = np.random.default_rng(dim * 2000 + n * 10 + dup * 2 + integer)
    sp = _random_spectrum(rng, dim, n, dup=dup, integer=integer, label="m")
    sorted_sp = _legacy_sort_signals(sp)  # legacy merge requires sorted input
    new = sorted_sp.merge_signals()
    old = _legacy_merge_signals(sorted_sp)
    _assert_spectra_equal(new, old)


def test_merge_signals_handles_unsorted_input():
    # Superset of the legacy contract: the new implementation sorts itself.
    sp = Spectrum_1D([3.0, 1.0, 3.0, 1.0], [1.0, 2.0, 3.0, 4.0], label="u")
    out = sp.merge_signals()
    np.testing.assert_array_equal(np.asarray(out.positions), [[1.0, 3.0]])
    np.testing.assert_allclose(np.asarray(out.intensities), [6.0, 4.0])
    assert out.label == "u"


def test_merge_signals_empty():
    sp = Spectrum(np.zeros((2, 0)), np.zeros(0), label="e")
    out = sp.merge_signals()
    assert type(out) is Spectrum
    assert out.label == "e"
    assert np.asarray(out.positions).shape == (2, 0)
    assert len(np.asarray(out.intensities)) == 0


# ---------------------------------------------------------------------------
# __add__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim,n,dup,integer", PARAMS)
def test_add_matches_legacy(dim, n, dup, integer):
    rng = np.random.default_rng(dim * 3000 + n * 10 + dup * 2 + integer)
    a = _random_spectrum(rng, dim, n, dup=dup, integer=integer, label="a")
    b = _random_spectrum(rng, dim, max(1, n - 2), dup=dup, integer=integer, label="b")
    _assert_spectra_equal(a + b, _legacy_add(a, b))


@pytest.mark.parametrize(
    "la,lb,expected",
    [(None, None, None), ("a", None, "a + None"), ("a", "b", "a + b")],
)
def test_add_label_handling(la, lb, expected):
    a = Spectrum_1D([1.0], [1.0], label=la)
    b = Spectrum_1D([2.0], [1.0], label=lb)
    assert (a + b).label == expected
    assert (a + b).label == _legacy_add(a, b).label


def test_add_merges_shared_positions():
    a = Spectrum_1D([1.0, 2.0], [1.0, 2.0], label="a")
    b = Spectrum_1D([2.0, 3.0], [5.0, 7.0], label="b")
    out = a + b
    np.testing.assert_array_equal(np.asarray(out.positions), [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(np.asarray(out.intensities), [1.0, 7.0, 7.0])


def test_add_with_empty():
    a = Spectrum_1D([1.0, 1.0, 2.0], [1.0, 2.0, 3.0], label="a")
    e = Spectrum(np.zeros((1, 0)), np.zeros(0), label=None)
    out = a + e
    # Duplicates within a single operand are merged too (legacy behaviour).
    np.testing.assert_array_equal(np.asarray(out.positions), [[1.0, 2.0]])
    np.testing.assert_allclose(np.asarray(out.intensities), [3.0, 3.0])
    _assert_spectra_equal(out, _legacy_add(a, e))


# ---------------------------------------------------------------------------
# Bookkeeping preserved: copy / pickle round-trips of results
# ---------------------------------------------------------------------------


def test_result_copy_and_pickle_roundtrip():
    import pickle

    rng = np.random.default_rng(99)
    sp = _random_spectrum(rng, 2, 20, dup=True, label="rt")
    out = (sp + sp).merge_signals().sort_signals()
    assert type(out) is Spectrum
    c = out.copy()
    _assert_spectra_equal(c, out)
    p = pickle.loads(pickle.dumps(out))
    _assert_spectra_equal(p, out)


# ---------------------------------------------------------------------------
# Timing sanity (informational print, no assert)
# ---------------------------------------------------------------------------


def test_timing_sanity_50k_print():
    rng = np.random.default_rng(7)
    n = 50_000
    sp = _random_spectrum(rng, 2, n, dup=True, label="big")
    sorted_sp = sp.sort_signals()

    t0 = time.perf_counter()
    new_merged = sorted_sp.merge_signals()
    t_new = time.perf_counter() - t0

    t0 = time.perf_counter()
    old_merged = _legacy_merge_signals(sorted_sp)
    t_old = time.perf_counter() - t0

    _assert_spectra_equal(new_merged, old_merged, rtol=1e-9)
    print(
        f"\nmerge_signals on {n} peaks: C++ {t_new * 1e3:.1f} ms vs "
        f"legacy Python {t_old * 1e3:.1f} ms "
        f"({t_old / max(t_new, 1e-9):.0f}x speedup)"
    )
