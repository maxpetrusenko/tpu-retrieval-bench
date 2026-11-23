"""Hypothesis property tests: exact variants vs numpy_ref under f32."""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from _topk_check import VARIANTS, check_matches_ref, check_structure, full_scores, run
from trbench import data
from trbench.reference import numpy_ref

SETTINGS = settings(
    max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow], derandomize=True
)


@st.composite
def problems(draw):
    n = draw(st.integers(1, 300))
    return {
        "m": draw(st.integers(1, 4)),
        "n": n,
        "d": draw(st.integers(1, 40)),
        "k": draw(st.integers(1, n)),
        "metric": draw(st.sampled_from(["ip", "l2", "cos"])),
        "mask_frac": draw(st.sampled_from([0.0, 0.0, 0.3, 0.9])),
        "seed": draw(st.integers(0, 2**16)),
    }


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@SETTINGS
@given(p=problems())
def test_exact_matches_ref_and_invariants(variant, p):
    q, e, mask = data.synthetic(p["m"], p["n"], p["d"], seed=p["seed"], mask_frac=p["mask_frac"])
    rs, ri = numpy_ref(q, e, p["k"], metric=p["metric"], mask=mask)
    s, i = run(variant, q, e, p["k"], p["metric"], mask)
    check_structure(variant, s, i, p["n"], mask)  # sorted desc, unique, in range, no masked ids
    check_matches_ref(s, i, rs, ri, full_scores(q, e, p["metric"], mask), p["k"])
