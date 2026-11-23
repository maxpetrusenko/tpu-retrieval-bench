"""V1: stock XLA baseline. Full Q @ E.T then lax.top_k.

prepare (db cast to compute dtype, norms) is a separate step so timing covers search only.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.typing import DTypeLike

from trbench._common import DenseIndex, JitFn, dense_scores, mark_invalid, prepare_dense


def search_impl(
    queries: Array, idx: DenseIndex, *, k: int, metric: str, compute_dtype: DTypeLike
) -> tuple[Array, Array]:
    s = dense_scores(queries, idx, metric=metric, compute_dtype=compute_dtype)
    vals, ids = lax.top_k(s, k)
    return mark_invalid(vals, ids.astype(jnp.int32))


def make_jit_topk_parts(
    k: int, *, metric: str = "ip", compute_dtype: DTypeLike = jnp.bfloat16, has_mask: bool = False
) -> tuple[JitFn, JitFn]:
    """(prepare(db[, mask]) -> index, search(queries, index) -> (scores, ids)), both jitted."""
    kw: dict[str, Any] = {"metric": metric, "compute_dtype": compute_dtype}
    if has_mask:
        prep_fn = jax.jit(lambda e, m: prepare_dense(e, m, **kw))
    else:
        prep_fn = jax.jit(lambda e: prepare_dense(e, None, **kw))
    return prep_fn, jax.jit(lambda q, idx: search_impl(q, idx, k=k, **kw))


def make_jit_topk(
    k: int, *, metric: str = "ip", compute_dtype: DTypeLike = jnp.bfloat16, has_mask: bool = False
) -> JitFn:
    """Return a jitted fn (queries, db[, mask]) -> (scores, ids) with static params bound."""
    kw: dict[str, Any] = {"metric": metric, "compute_dtype": compute_dtype}
    if has_mask:
        return jax.jit(lambda q, e, m: search_impl(q, prepare_dense(e, m, **kw), k=k, **kw))
    return jax.jit(lambda q, e: search_impl(q, prepare_dense(e, None, **kw), k=k, **kw))


def jit_topk(
    queries: Array,
    db: Array,
    k: int,
    *,
    metric: str = "ip",
    mask: Array | None = None,
    compute_dtype: DTypeLike = jnp.bfloat16,
) -> tuple[Array, Array]:
    f = make_jit_topk(k, metric=metric, compute_dtype=compute_dtype, has_mask=mask is not None)
    out: tuple[Array, Array] = f(queries, db) if mask is None else f(queries, db, mask)
    return out
