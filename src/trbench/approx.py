"""A1: lax.approx_max_k (approximate; reported separately from exact variants)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from trbench._common import dense_scores, prepare_dense


def approx_search(queries, idx, *, k, metric, compute_dtype, recall_target):
    s = dense_scores(queries, idx, metric=metric, compute_dtype=compute_dtype)
    v, i = lax.approx_max_k(s, k, recall_target=recall_target)
    return v, i.astype(jnp.int32)


def make_approx_parts(k, *, recall_target=0.95, metric="ip", compute_dtype=jnp.bfloat16):
    kw = {"metric": metric, "compute_dtype": compute_dtype}
    return (jax.jit(lambda e: prepare_dense(e, None, **kw)),
            jax.jit(lambda q, idx: approx_search(q, idx, k=k, recall_target=recall_target, **kw)))


def make_approx(k, *, recall_target=0.95, metric="ip", compute_dtype=jnp.bfloat16,
                has_mask=False):
    kw = {"metric": metric, "compute_dtype": compute_dtype}
    sk = {"k": k, "recall_target": recall_target, **kw}
    if has_mask:
        return jax.jit(lambda q, e, m: approx_search(q, prepare_dense(e, m, **kw), **sk))
    return jax.jit(lambda q, e: approx_search(q, prepare_dense(e, None, **kw), **sk))


def approx_topk(queries, db, k, *, recall_target=0.95, metric="ip", mask=None,
                compute_dtype=jnp.bfloat16):
    f = make_approx(k, recall_target=recall_target, metric=metric,
                    compute_dtype=compute_dtype, has_mask=mask is not None)
    return f(queries, db) if mask is None else f(queries, db, mask)
