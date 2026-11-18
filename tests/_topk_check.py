"""Shared helpers for exact-variant hardening tests (grid, property, padding, ties).

Grid/parametrize style follows jax tests/pallas (pytest.mark.parametrize shape x k product);
padding/-inf-tail invariants follow the idea in vllm-project/tpu-inference kernel tests
(Apache-2.0, ideas only, no code copied).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from trbench.blockwise import blockwise
from trbench.jit_topk import jit_topk
from trbench.pallas_score import pallas_score
from trbench.reference import scores_ref

F32 = jnp.float32
TOL = 1e-4


def _jit(q, e, k, metric, mask):
    return jit_topk(q, e, k, metric=metric, mask=mask, compute_dtype=F32)


def _blockwise(q, e, k, metric, mask):
    return blockwise(q, e, k, metric=metric, mask=mask, compute_dtype=F32, block_size=64)


def _pallas(q, e, k, metric, mask):
    # small blocks force multi-tile grids, multi-chunk scans and D tiling; interpret on CPU
    return pallas_score(q, e, k, metric=metric, mask=mask, compute_dtype=F32,
                        bm=16, bn=128, bk=128, chunk=256)


VARIANTS = {"jit_topk": _jit, "blockwise": _blockwise, "pallas": _pallas}


def run(name, q, e, k, metric="ip", mask=None):
    s, i = VARIANTS[name](q, e, k, metric, mask)
    return np.asarray(s), np.asarray(i)


def check_structure(name, s, i, n, mask=None):
    """Sorted desc, ids unique and in range, padding only at the row tail, masked never returned."""
    assert s.shape == i.shape
    assert np.all(s[:, :-1] >= s[:, 1:]), "scores not sorted descending"
    assert np.all(i >= -1) and np.all(i < n), "id out of range"
    for row_s, row_i in zip(s, i, strict=True):
        real = row_i[row_i >= 0]
        assert len(set(real.tolist())) == len(real), "duplicate ids"
        pad = ~np.isfinite(row_s)
        if pad.any():
            first = int(np.argmax(pad))
            assert pad[first:].all(), "-inf score before a finite one"
        assert np.array_equal(row_i == -1, pad), "-1 ids must sit exactly at the -inf tail"
    if mask is not None:
        fin = np.isfinite(s)
        got = i[fin]
        assert np.asarray(mask)[got].all(), "masked id returned with finite score"


def check_matches_ref(s, i, ref_s, ref_i, full, k):
    """Set-of-ids equality with numpy_ref, tolerant only for ids tied at the k-th boundary."""
    valid = np.isfinite(ref_s)
    assert np.array_equal(np.isfinite(s), valid), "different number of valid results"
    np.testing.assert_allclose(np.where(valid, s, 0), np.where(valid, ref_s, 0),
                               rtol=TOL, atol=TOL)
    for r in range(s.shape[0]):
        nv = int(valid[r].sum())
        got, want = set(i[r, :nv].tolist()), set(ref_i[r, :nv].tolist())
        if got == want:
            continue
        kth = ref_s[r, nv - 1]
        diff = (got ^ want)
        slack = TOL * max(abs(float(kth)), 1.0)
        assert all(abs(full[r, j] - kth) <= slack for j in diff), (
            f"row {r}: id sets differ away from the k-th boundary: {sorted(diff)[:8]}")


def full_scores(q, e, metric, mask):
    return scores_ref(q, e, metric=metric, mask=mask)
