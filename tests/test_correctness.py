import jax.numpy as jnp
import numpy as np
import pytest

from trbench import data
from trbench.jit_topk import jit_topk
from trbench.reference import numpy_ref, scores_ref

M, N, D = 8, 1000, 100  # N not a multiple of 128


def _case(metric, use_mask):
    q, e, mask = data.synthetic(M, N, D, seed=1, mask_frac=0.3 if use_mask else 0.0)
    return q, e, mask


@pytest.mark.parametrize("metric", ["ip", "l2", "cos"])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("k", [1, 10, 100])
def test_f32_exact_ids(metric, use_mask, k):
    q, e, mask = _case(metric, use_mask)
    rs, ri = numpy_ref(q, e, k, metric=metric, mask=mask)
    s, i = jit_topk(q, e, k, metric=metric, mask=mask, compute_dtype=jnp.float32)
    np.testing.assert_array_equal(np.asarray(i), ri)
    np.testing.assert_allclose(np.asarray(s), rs, rtol=1e-4, atol=1e-4)
    if mask is not None:
        assert mask[np.asarray(i)].all()


@pytest.mark.parametrize("metric", ["ip", "l2", "cos"])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("k", [1, 10, 100])
def test_bf16_recall_and_scores(metric, use_mask, k):
    q, e, mask = _case(metric, use_mask)
    rs, _ = numpy_ref(q, e, k, metric=metric, mask=mask)
    s, i = jit_topk(q, e, k, metric=metric, mask=mask, compute_dtype=jnp.bfloat16)
    s, i = np.asarray(s), np.asarray(i)
    full = scores_ref(q, e, metric=metric, mask=mask)
    # returned ids' true scores must be close to reported bf16 scores
    true = np.take_along_axis(full, i, axis=1)
    scale = np.abs(rs).max()
    # ann-benchmarks distance-threshold recall on the true scores of returned ids (bf16 near-ties)
    recall = np.mean(np.sum(true >= rs[:, -1:] - 0.02 * scale, axis=1) / k)
    assert recall >= 0.99
    np.testing.assert_allclose(s, true, atol=0.05 * scale)
    np.testing.assert_allclose(s[:, 0], rs[:, 0], atol=0.05 * scale)


def test_tie_policy_lower_id_wins():
    db = np.ones((10, 4), dtype=np.float32)
    q = np.ones((1, 4), dtype=np.float32)
    _, ids = numpy_ref(q, db, 3)
    assert ids[0].tolist() == [0, 1, 2]
    _, ids = jit_topk(q, db, 3, compute_dtype=jnp.float32)
    assert np.asarray(ids)[0].tolist() == [0, 1, 2]
