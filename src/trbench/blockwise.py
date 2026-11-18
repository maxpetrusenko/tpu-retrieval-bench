"""V2: scan over db blocks, per-block top-k, running merge. Never materializes M x N.

Also stage-split helpers for R5: score only (full M x N scores) and select only (top_k on scores).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from trbench._common import block_topk, init_running, merge_topk, prep

DEFAULT_BLOCK = 65536


def blockwise_impl(queries, db, mask, *, k, metric, compute_dtype, block_size):
    m, n = queries.shape[0], db.shape[0]
    b = min(block_size, n)
    nb = -(-n // b)
    npad = nb * b - n
    q, e = prep(queries, db, metric)
    valid = jnp.ones((n,), bool) if mask is None else mask.astype(bool)
    if npad:
        e = jnp.pad(e, ((0, npad), (0, 0)))
        valid = jnp.pad(valid, (0, npad))  # padded rows are masked out
    eb = e.reshape(nb, b, -1)
    vb = valid.reshape(nb, b)
    qc = q.astype(compute_dtype)

    def step(carry, xs):
        run_v, run_i = carry
        e_blk, v_blk, off = xs
        s = jnp.matmul(qc, e_blk.astype(compute_dtype).T, preferred_element_type=jnp.float32)
        if metric == "l2":
            s = 2.0 * s - jnp.sum(e_blk * e_blk, axis=1)[None, :]
        s = jnp.where(v_blk[None, :], s, -jnp.inf)
        bv, bi = block_topk(s, k, off)
        return merge_topk(run_v, run_i, bv, bi, k), None

    offs = jnp.arange(nb, dtype=jnp.int32) * b
    (vals, ids), _ = lax.scan(step, init_running(m, k), (eb, vb, offs))
    return vals, ids


def make_blockwise(k, *, metric="ip", compute_dtype=jnp.bfloat16, has_mask=False,
                   block_size=DEFAULT_BLOCK):
    kw = {"k": k, "metric": metric, "compute_dtype": compute_dtype, "block_size": block_size}
    if has_mask:
        return jax.jit(lambda q, e, m: blockwise_impl(q, e, m, **kw))
    return jax.jit(lambda q, e: blockwise_impl(q, e, None, **kw))


def blockwise(queries, db, k, *, metric="ip", mask=None, compute_dtype=jnp.bfloat16,
              block_size=DEFAULT_BLOCK):
    f = make_blockwise(k, metric=metric, compute_dtype=compute_dtype,
                       has_mask=mask is not None, block_size=block_size)
    return f(queries, db) if mask is None else f(queries, db, mask)


def score_impl(queries, db, mask, *, metric, compute_dtype):
    q, e = prep(queries, db, metric)
    s = jnp.matmul(q.astype(compute_dtype), e.astype(compute_dtype).T,
                   preferred_element_type=jnp.float32)
    if metric == "l2":
        s = 2.0 * s - jnp.sum(e * e, axis=1)[None, :]
    if mask is not None:
        s = jnp.where(mask[None, :], s, -jnp.inf)
    return s


def make_score_only(*, metric="ip", compute_dtype=jnp.bfloat16, has_mask=False):
    """Scoring stage of V1 alone: (queries, db[, mask]) -> full (M, N) f32 scores."""
    fn = partial(score_impl, metric=metric, compute_dtype=compute_dtype)
    if has_mask:
        return jax.jit(lambda q, e, m: fn(q, e, m))
    return jax.jit(lambda q, e: fn(q, e, None))


def make_select_only(k):
    """Selection stage alone: precomputed (M, N) scores -> (vals, ids)."""
    def f(scores):
        v, i = lax.top_k(scores, k)
        return v, i.astype(jnp.int32)
    return jax.jit(f)
