import jax.numpy as jnp
import numpy as np
import pytest

from trbench import data
from trbench.approx import approx_topk
from trbench.blockwise import blockwise, make_score_only, make_select_only
from trbench.reference import numpy_ref, scores_ref

M, N, D = 8, 1000, 100


@pytest.mark.parametrize("metric", ["ip", "l2", "cos"])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("k", [1, 10, 100])
@pytest.mark.parametrize("block", [256, 333, 4096])  # non-divisor, and block > N
def test_f32_exact_ids(metric, use_mask, k, block):
    q, e, mask = data.synthetic(M, N, D, seed=1, mask_frac=0.3 if use_mask else 0.0)
    rs, ri = numpy_ref(q, e, k, metric=metric, mask=mask)
    s, i = blockwise(q, e, k, metric=metric, mask=mask, compute_dtype=jnp.float32,
                     block_size=block)
    np.testing.assert_array_equal(np.asarray(i), ri)
    np.testing.assert_allclose(np.asarray(s), rs, rtol=1e-4, atol=1e-4)


def test_bf16_recall():
    q, e, mask = data.synthetic(M, N, D, seed=2, mask_frac=0.3)
    rs, _ = numpy_ref(q, e, 10, mask=mask)
    s, _ = blockwise(q, e, 10, mask=mask, block_size=256)
    assert data.recall_at_k(rs, np.asarray(s), eps=1e-2) >= 0.99


def test_tie_policy_lower_id_wins():
    db = np.ones((10, 4), dtype=np.float32)
    q = np.ones((1, 4), dtype=np.float32)
    _, ids = blockwise(q, db, 3, compute_dtype=jnp.float32, block_size=4)
    assert np.asarray(ids)[0].tolist() == [0, 1, 2]


def test_stage_split_matches():
    q, e, mask = data.synthetic(M, N, D, seed=3, mask_frac=0.2)
    sc = make_score_only(metric="l2", compute_dtype=jnp.float32, has_mask=True)(q, e, mask)
    np.testing.assert_allclose(
        np.asarray(sc), scores_ref(q, e, metric="l2", mask=mask).astype(np.float32),
        rtol=1e-4, atol=1e-3)
    _, i = make_select_only(10)(sc)
    _, ri = numpy_ref(q, e, 10, metric="l2", mask=mask)
    np.testing.assert_array_equal(np.asarray(i), ri)


@pytest.mark.parametrize("target", [0.9, 0.95, 0.99])
def test_approx_recall(target):
    q, e, _ = data.synthetic(16, 4096, 64, seed=4)
    rs, _ = numpy_ref(q, e, 10)
    s, _ = approx_topk(q, e, 10, recall_target=target, compute_dtype=jnp.float32)
    assert data.recall_at_k(rs, np.asarray(s), eps=1e-4) >= target - 0.05
