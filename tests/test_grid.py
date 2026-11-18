"""Shape x k grid with non-aligned N and D for every exact variant (f32, CPU/interpret)."""

import numpy as np
import pytest

from _topk_check import (
    VARIANTS,
    check_matches_ref,
    check_structure,
    full_scores,
    run,
)
from trbench import data
from trbench.reference import numpy_ref

NS = [1, 7, 100, 129, 1003]
DS = [1, 7, 100, 129]
KS = [1, 10, 100, 128, "N"]
M = 3


def _cases():
    for n in NS:
        for d in DS:
            for kk in KS:
                k = n if kk == "N" else kk
                if k > n:
                    continue
                yield n, d, k


CASES = sorted(set(_cases()))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("n,d,k", CASES)
def test_grid_ip_matches_ref(variant, n, d, k):
    q, e, _ = data.synthetic(M, n, d, seed=n * 31 + d)
    rs, ri = numpy_ref(q, e, k)
    s, i = run(variant, q, e, k)
    check_structure(variant, s, i, n)
    check_matches_ref(s, i, rs, ri, full_scores(q, e, "ip", None), k)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("metric", ["l2", "cos"])
@pytest.mark.parametrize("n,d,k", [(7, 7, 7), (129, 100, 10), (1003, 129, 128), (100, 1, 100)])
def test_grid_other_metrics_with_mask(variant, metric, n, d, k):
    q, e, mask = data.synthetic(M, n, d, seed=n + d, mask_frac=0.4)
    rs, ri = numpy_ref(q, e, k, metric=metric, mask=mask)
    s, i = run(variant, q, e, k, metric, mask)
    check_structure(variant, s, i, n, mask)
    check_matches_ref(s, i, rs, ri, full_scores(q, e, metric, mask), k)


def test_grid_covers_required_k_values():
    ks = {k for _, _, k in CASES}
    assert {1, 10, 100, 128} <= ks
    assert any(k == n for n, _, k in CASES if n > 1)
    assert np.isin([1, 7, 100, 129, 1003], [n for n, _, _ in CASES]).all()
