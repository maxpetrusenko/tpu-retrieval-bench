"""Tie-heavy and worst-case inputs; documented tie policy (lower id wins in numpy_ref)."""

import numpy as np
import pytest

from _topk_check import VARIANTS, check_structure, run
from trbench import data
from trbench.reference import numpy_ref, scores_ref


def _tie_cases():
    for mode in ("duplicate", "quantized"):
        yield f"tie_{mode}", data.tie_heavy(4, 301, 9, seed=3, mode=mode)
        yield f"tie_{mode}_mask", data.tie_heavy(4, 301, 9, seed=4, mode=mode, mask_frac=0.3)
    for kind in ("all_equal", "ascending", "descending"):
        yield f"worst_{kind}", data.worst_case(3, 260, 5, kind=kind)


CASES = dict(_tie_cases())


def test_data_modes_really_tie():
    q, e, _ = CASES["tie_duplicate"]
    assert len(np.unique(e, axis=0)) <= 7
    q, e, _ = CASES["tie_quantized"]
    s = scores_ref(q, e)
    assert len(np.unique(s[0])) < s.shape[1] // 3
    q, e, _ = CASES["worst_all_equal"]
    s = scores_ref(q, e)
    assert (s == s[:, :1]).all()


def test_data_modes_reject_unknown():
    with pytest.raises(ValueError):
        data.tie_heavy(2, 4, 4, mode="nope")
    with pytest.raises(ValueError):
        data.worst_case(2, 4, 4, kind="nope")


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("k", [1, 10, 100])
def test_reference_tie_policy_lower_id_wins(name, k):
    q, e, mask = CASES[name]
    s, i = numpy_ref(q, e, k, mask=mask)
    full = scores_ref(q, e, mask=mask)
    for r in range(q.shape[0]):
        # within every run of equal scores, ids ascend (stable sort, lower id first)
        for a in range(k - 1):
            if s[r, a] == s[r, a + 1] and np.isfinite(s[r, a]):
                assert i[r, a] < i[r, a + 1]
        # and the returned set is the lexicographic (score desc, id asc) top-k
        order = sorted(range(full.shape[1]), key=lambda j: (-full[r, j], j))[:k]
        assert i[r].tolist() == order


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("k", [1, 10, 100])
@pytest.mark.parametrize("metric", ["ip", "l2"])
def test_variants_tie_set_equality(variant, name, k, metric):
    """Scores equal the reference exactly; ids differ from it only among tied candidates."""
    q, e, mask = CASES[name]
    rs, _ = numpy_ref(q, e, k, metric=metric, mask=mask)
    full = scores_ref(q, e, metric=metric, mask=mask)
    s, i = run(variant, q, e, k, metric, mask)
    check_structure(variant, s, i, e.shape[0], mask)
    np.testing.assert_array_equal(s, rs)  # integer-valued data: f32 scores are exact
    for r in range(q.shape[0]):
        nv = int(np.isfinite(rs[r]).sum())
        got = i[r, :nv]
        # each returned id carries the score reported at its slot
        np.testing.assert_array_equal(full[r, got], rs[r, :nv].astype(np.float64))
        kth = rs[r, nv - 1]
        above = set(np.flatnonzero(full[r] > kth).tolist())
        assert above <= set(got.tolist()), "an id strictly above the k-th score was dropped"


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("name", ["worst_all_equal", "tie_duplicate", "tie_quantized"])
def test_variants_follow_lower_id_policy(variant, name):
    q, e, mask = CASES[name]
    _, ri = numpy_ref(q, e, 10, mask=mask)
    _, i = run(variant, q, e, 10, "ip", mask)
    np.testing.assert_array_equal(i, ri)
