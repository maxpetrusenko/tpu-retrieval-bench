"""V1: stock XLA baseline. Full Q @ E.T then lax.top_k.

prepare (db cast to compute dtype, norms) is a separate step so timing covers search only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from trbench._common import dense_scores, mark_invalid, prepare_dense


def search_impl(queries, idx, *, k, metric, compute_dtype):
    s = dense_scores(queries, idx, metric=metric, compute_dtype=compute_dtype)
    vals, ids = lax.top_k(s, k)
    return mark_invalid(vals, ids.astype(jnp.int32))


def make_jit_topk_parts(k, *, metric="ip", compute_dtype=jnp.bfloat16, has_mask=False):
    """(prepare(db[, mask]) -> index, search(queries, index) -> (scores, ids)), both jitted."""
    kw = {"metric": metric, "compute_dtype": compute_dtype}
    if has_mask:
        prep_fn = jax.jit(lambda e, m: prepare_dense(e, m, **kw))
    else:
        prep_fn = jax.jit(lambda e: prepare_dense(e, None, **kw))
    return prep_fn, jax.jit(lambda q, idx: search_impl(q, idx, k=k, **kw))


def make_jit_topk(k, *, metric="ip", compute_dtype=jnp.bfloat16, has_mask=False):
    """Return a jitted fn (queries, db[, mask]) -> (scores, ids) with static params bound."""
    kw = {"metric": metric, "compute_dtype": compute_dtype}
    if has_mask:
        return jax.jit(lambda q, e, m: search_impl(q, prepare_dense(e, m, **kw), k=k, **kw))
    return jax.jit(lambda q, e: search_impl(q, prepare_dense(e, None, **kw), k=k, **kw))


def jit_topk(queries, db, k, *, metric="ip", mask=None, compute_dtype=jnp.bfloat16):
    f = make_jit_topk(k, metric=metric, compute_dtype=compute_dtype, has_mask=mask is not None)
    if mask is None:
        return f(queries, db)
    return f(queries, db, mask)
