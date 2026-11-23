import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trbench import data
from trbench.pallas_fused import (
    make_fused_parts,
    make_pallas_fused,
    pallas_fused_bucket,
    pallas_fused_topk,
)
from trbench.reference import numpy_ref

UNSUPPORTED_PLATFORM = (
    "unknown platform",
    "unsupported platform",
    "not supported on",
    "no lowering rule registered for platform",
    "platform tpu",
)
BLK = {"bm": 16, "bn": 128, "bk": 128, "chunk": 256, "compute_dtype": jnp.float32}


@pytest.mark.parametrize("m,n,d", [(8, 600, 100), (5, 300, 130), (16, 128, 128)])
@pytest.mark.parametrize("metric", ["ip", "l2", "cos"])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("k", [1, 10, 100, 128])
def test_f32_exact_ids(m, n, d, metric, use_mask, k):
    q, e, mask = data.synthetic(m, n, d, seed=1, mask_frac=0.3 if use_mask else 0.0)
    rs, ri = numpy_ref(q, e, k, metric=metric, mask=mask)
    s, i = pallas_fused_topk(q, e, k, metric=metric, mask=mask, **BLK)
    kk = min(k, n)
    np.testing.assert_array_equal(np.asarray(i)[:, :kk], ri[:, :kk])
    np.testing.assert_allclose(np.asarray(s)[:, :kk], rs[:, :kk], rtol=1e-4, atol=1e-4)
    if k > n:  # -1 tail like the other variants
        assert (np.asarray(i)[:, n:] == -1).all()
        assert np.isneginf(np.asarray(s)[:, n:]).all()


def test_heavy_mask_inf_ids_match_oracle():
    q, e, mask = data.synthetic(4, 400, 64, seed=5, mask_frac=0.97)
    rs, ri = numpy_ref(q, e, 50, mask=mask)  # fewer than 50 valid: -inf tail with id -1
    s, i = pallas_fused_topk(q, e, 50, mask=mask, **BLK)
    np.testing.assert_array_equal(np.asarray(i), ri)
    np.testing.assert_array_equal(np.isneginf(np.asarray(s)), np.isneginf(rs))


def test_tie_policy_across_tiles():
    db = np.ones((300, 4), dtype=np.float32)
    q = np.ones((1, 4), dtype=np.float32)
    _, ids = pallas_fused_topk(
        q, db, 5, compute_dtype=jnp.float32, bm=16, bn=128, bk=128, chunk=256
    )
    assert np.asarray(ids)[0].tolist() == [0, 1, 2, 3, 4]
    db[:, 0] = 0.0
    db[[200, 3, 130]] = 2.0  # equal winners in three different tiles: ascending ids
    _, ids = pallas_fused_topk(
        q, db, 3, compute_dtype=jnp.float32, bm=16, bn=128, bk=128, chunk=256
    )
    assert np.asarray(ids)[0].tolist() == [3, 130, 200]


def test_kprime_below_k_is_approximate_but_valid():
    q, e, _ = data.synthetic(8, 600, 100, seed=2)
    rs, _ = numpy_ref(q, e, 10)
    s, i = pallas_fused_topk(q, e, 10, kprime=3, **BLK)
    assert np.asarray(s).shape == (8, 10)
    assert (np.asarray(i) >= 0).all()
    assert data.recall_at_k(rs, np.asarray(s), eps=1e-4) >= 0.5


def test_bad_args():
    with pytest.raises(ValueError):
        make_fused_parts(129, 1000)
    with pytest.raises(ValueError):
        make_fused_parts(10, 1000, kprime=11)
    with pytest.raises(ValueError):
        make_fused_parts(10, 1000, mode="bucket", lanes=100)


def test_bucket_is_valid_and_recall_measured():
    q, e, mask = data.synthetic(8, 600, 100, seed=3, mask_frac=0.2)
    rs, _ = numpy_ref(q, e, 10, mask=mask)
    s, i = pallas_fused_bucket(q, e, 10, mask=mask, **BLK)
    s, i = np.asarray(s), np.asarray(i)
    assert (i < 600).all()
    assert (s[:, :-1] >= s[:, 1:]).all()  # sorted desc
    assert data.recall_at_k(rs, s, eps=1e-4) >= 0.8  # 5 tiles x 128 bins: collisions are rare
    # bins >= n_tiles * k: every top-k element sits alone in its bin region, so recall is 1
    e_small = e[:128]
    rs2, _ = numpy_ref(q, e_small, 10)
    s2, _ = pallas_fused_bucket(q, e_small, 10, **BLK)
    assert data.recall_at_k(rs2, np.asarray(s2), eps=1e-4) >= 0.99


def test_prepared_parts_match_one_shot():
    q, e, mask = data.synthetic(8, 600, 100, seed=4, mask_frac=0.2)
    prep, search = make_fused_parts(10, 600, metric="l2", has_mask=True, **BLK)
    _, i1 = search(q, prep(e, mask))
    _, i2 = pallas_fused_topk(q, e, 10, metric="l2", mask=mask, **BLK)
    np.testing.assert_array_equal(np.asarray(i1), np.asarray(i2))


@pytest.mark.parametrize("mode", ["topk", "bucket"])
def test_tpu_lowering_from_cpu(mode):
    f = make_pallas_fused(
        10,
        512,
        metric="l2",
        has_mask=True,
        bm=16,
        bn=128,
        bk=128,
        chunk=256,
        interpret=False,
        mode=mode,
    )
    q = jax.ShapeDtypeStruct((16, 128), jnp.float32)
    e = jax.ShapeDtypeStruct((512, 128), jnp.float32)
    mk = jax.ShapeDtypeStruct((512,), jnp.bool_)
    try:
        text = f.trace(q, e, mk).lower(lowering_platforms=("tpu",)).as_text()
    except Exception as ex:
        msg = str(ex).lower()
        if any(t in msg for t in UNSUPPORTED_PLATFORM):
            pytest.skip(f"TPU lowering unsupported on this host: {str(ex)[:160]}")
        raise
    assert "tpu_custom_call" in text or "mosaic" in text.lower()
