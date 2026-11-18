"""A1: lax.approx_max_k (approximate; reported separately from exact variants)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

from trbench.blockwise import score_impl


def approx_impl(queries, db, mask, *, k, metric, compute_dtype, recall_target):
    s = score_impl(queries, db, mask, metric=metric, compute_dtype=compute_dtype)
    v, i = lax.approx_max_k(s, k, recall_target=recall_target)
    return v, i.astype(jnp.int32)


def make_approx(k, *, recall_target=0.95, metric="ip", compute_dtype=jnp.bfloat16,
                has_mask=False):
    kw = {"k": k, "metric": metric, "compute_dtype": compute_dtype,
          "recall_target": recall_target}
    if has_mask:
        return jax.jit(lambda q, e, m: approx_impl(q, e, m, **kw))
    return jax.jit(lambda q, e: approx_impl(q, e, None, **kw))


def approx_topk(queries, db, k, *, recall_target=0.95, metric="ip", mask=None,
                compute_dtype=jnp.bfloat16):
    f = make_approx(k, recall_target=recall_target, metric=metric,
                    compute_dtype=compute_dtype, has_mask=mask is not None)
    return f(queries, db) if mask is None else f(queries, db, mask)
