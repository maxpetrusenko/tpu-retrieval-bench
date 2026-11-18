"""Padding and mask invariants when k exceeds the number of valid candidates."""

import numpy as np
import pytest

from _topk_check import VARIANTS, check_structure, run
from trbench import data
from trbench.approx import approx_topk
from trbench.reference import numpy_ref


def _masked(n, valid_ids):
    mask = np.zeros(n, dtype=bool)
    mask[list(valid_ids)] = True
    return mask


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("metric", ["ip", "l2", "cos"])
@pytest.mark.parametrize("n,valid,k", [(20, 3, 6), (300, 5, 10), (300, 1, 128), (129, 0, 4)])
def test_pad_only_at_tail(variant, metric, n, valid, k):
    q, e, _ = data.synthetic(3, n, 16, seed=n + valid)
    rng = np.random.default_rng(valid)
    mask = _masked(n, rng.choice(n, size=valid, replace=False))
    s, i = run(variant, q, e, k, metric, mask)
    check_structure(variant, s, i, n, mask)
    assert np.isfinite(s).sum(axis=1).tolist() == [valid] * 3
    assert np.all(np.isneginf(s[:, valid:]))
    assert np.all(i[:, valid:] == -1)
    assert np.all(i[:, :valid] >= 0)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_masked_ids_never_returned_at_finite_score(variant):
    q, e, mask = data.synthetic(4, 257, 12, seed=5, mask_frac=0.7)
    s, i = run(variant, q, e, 50, "ip", mask)
    assert mask[i[np.isfinite(s)]].all()


def test_reference_pads_with_minus_one_at_neg_inf():
    q, e, _ = data.synthetic(2, 20, 8, seed=0)
    mask = _masked(20, [3, 5, 9])
    s, i = numpy_ref(q, e, 6, mask=mask)
    assert np.all(np.isneginf(s[:, 3:]))
    assert np.all(i[:, 3:] == -1)
    assert mask[i[:, :3]].all()


def test_approx_never_returns_masked_with_finite_score():
    q, e, mask = data.synthetic(3, 500, 16, seed=2, mask_frac=0.5)
    s, i = approx_topk(q, e, 10, mask=mask)
    s, i = np.asarray(s), np.asarray(i)
    fin = np.isfinite(s)
    assert fin.any()
    assert mask[i[fin]].all()


@pytest.mark.parametrize("n,valid,k", [(20, 3, 6), (300, 1, 16), (129, 0, 4)])
def test_approx_pads_with_minus_one_at_neg_inf(n, valid, k):
    q, e, _ = data.synthetic(3, n, 16, seed=n)
    mask = _masked(n, np.random.default_rng(valid).choice(n, size=valid, replace=False))
    s, i = approx_topk(q, e, k, mask=mask)
    s, i = np.asarray(s), np.asarray(i)
    assert np.all(np.isneginf(s[:, valid:]))
    assert np.all(i[:, valid:] == -1)
    assert np.all(i[:, :valid] >= 0)
