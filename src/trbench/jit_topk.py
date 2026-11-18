"""V1: stock XLA baseline. Full Q @ E.T then lax.top_k."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def _l2_normalize(x):
    n = jnp.linalg.norm(x, axis=1, keepdims=True)
    return x / jnp.maximum(n, 1e-30)


def jit_topk_impl(queries, db, mask, *, k, metric, compute_dtype):
    q = queries.astype(jnp.float32)
    e = db.astype(jnp.float32)
    if metric == "cos":
        q, e = _l2_normalize(q), _l2_normalize(e)
    s = jnp.matmul(
        q.astype(compute_dtype),
        e.astype(compute_dtype).T,
        preferred_element_type=jnp.float32,
    )
    if metric == "l2":
        s = 2.0 * s - jnp.sum(e * e, axis=1)[None, :]
    if mask is not None:
        s = jnp.where(mask[None, :], s, -jnp.inf)
    vals, ids = lax.top_k(s, k)
    return vals, ids.astype(jnp.int32)


def make_jit_topk(k, *, metric="ip", compute_dtype=jnp.bfloat16, has_mask=False):
    """Return a jitted fn (queries, db[, mask]) -> (scores, ids) with static params bound."""
    if has_mask:
        fn = partial(jit_topk_impl, k=k, metric=metric, compute_dtype=compute_dtype)
        return jax.jit(lambda q, e, m: fn(q, e, m))
    fn = partial(jit_topk_impl, mask=None, k=k, metric=metric, compute_dtype=compute_dtype)
    return jax.jit(lambda q, e: fn(q, e))


def jit_topk(queries, db, k, *, metric="ip", mask=None, compute_dtype=jnp.bfloat16):
    f = make_jit_topk(k, metric=metric, compute_dtype=compute_dtype, has_mask=mask is not None)
    if mask is None:
        return f(queries, db)
    return f(queries, db, mask)
