import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trbench import data
from trbench.pallas_score import make_pallas_parts, make_pallas_score, pallas_score
from trbench.reference import numpy_ref

UNSUPPORTED_PLATFORM = ("unknown platform", "unsupported platform", "not supported on",
                        "no lowering rule registered for platform", "platform tpu")
M, N, D = 8, 600, 100  # N, D non-multiples of 128


@pytest.mark.parametrize("metric", ["ip", "l2", "cos"])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("k", [1, 10, 100])
def test_f32_exact_ids(metric, use_mask, k):
    q, e, mask = data.synthetic(M, N, D, seed=1, mask_frac=0.3 if use_mask else 0.0)
    rs, ri = numpy_ref(q, e, k, metric=metric, mask=mask)
    # small blocks force multi-tile grid, multi-chunk scan and D tiling
    s, i = pallas_score(q, e, k, metric=metric, mask=mask, compute_dtype=jnp.float32,
                        bm=16, bn=128, bk=128, chunk=256)
    np.testing.assert_array_equal(np.asarray(i), ri)
    np.testing.assert_allclose(np.asarray(s), rs, rtol=1e-4, atol=1e-4)


def test_bf16_recall():
    q, e, mask = data.synthetic(M, N, D, seed=2, mask_frac=0.3)
    rs, _ = numpy_ref(q, e, 10, mask=mask)
    s, _ = pallas_score(q, e, 10, mask=mask, bm=16, bn=128, bk=128, chunk=256)
    assert data.recall_at_k(rs, np.asarray(s), eps=1e-2) >= 0.99


def test_tie_policy():
    db = np.ones((10, 4), dtype=np.float32)
    q = np.ones((1, 4), dtype=np.float32)
    _, ids = pallas_score(q, db, 3, compute_dtype=jnp.float32)
    assert np.asarray(ids)[0].tolist() == [0, 1, 2]


def test_tpu_lowering_from_cpu():
    f = make_pallas_score(10, metric="l2", has_mask=True, bm=16, bn=128, bk=128,
                          chunk=256, interpret=False)
    q = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    e = jax.ShapeDtypeStruct((512, 128), jnp.float32)
    mk = jax.ShapeDtypeStruct((512,), jnp.bool_)
    try:
        text = f.trace(q, e, mk).lower(lowering_platforms=("tpu",)).as_text()
    except Exception as ex:
        msg = str(ex).lower()
        # skip only when this host cannot lower for TPU at all; any other failure is a real bug
        if any(t in msg for t in UNSUPPORTED_PLATFORM):
            pytest.skip(f"TPU lowering unsupported on this host: {str(ex)[:160]}")
        raise
    assert "tpu_custom_call" in text or "mosaic" in text.lower()


def test_prepared_parts_match_one_shot_and_use_small_last_chunk():
    q, e, mask = data.synthetic(M, N, D, seed=3, mask_frac=0.2)
    kw = {"bm": 16, "bn": 128, "bk": 128, "chunk": 256, "compute_dtype": jnp.float32}
    prep, search = make_pallas_parts(10, metric="l2", has_mask=True, **kw)
    idx = prep(e, mask)
    assert idx.e_rem is not None and idx.e_rem.shape[0] == 128  # 640 = 2 * 256 + 128
    s1, i1 = search(q, idx)
    s2, i2 = pallas_score(q, e, 10, metric="l2", mask=mask, **kw)
    np.testing.assert_array_equal(np.asarray(i1), np.asarray(i2))
    np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))
