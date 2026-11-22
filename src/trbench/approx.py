"""A1: lax.approx_max_k (approximate; reported separately from exact variants)."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.typing import DTypeLike

from trbench._common import DenseIndex, JitFn, dense_scores, mark_invalid, prepare_dense


def approx_search(queries: Array, idx: DenseIndex, *, k: int, metric: str,
                  compute_dtype: DTypeLike, recall_target: float) -> tuple[Array, Array]:
    s = dense_scores(queries, idx, metric=metric, compute_dtype=compute_dtype)
    v, i = lax.approx_max_k(s, k, recall_target=recall_target)
    return mark_invalid(v, i.astype(jnp.int32))


def make_approx_parts(k: int, *, recall_target: float = 0.95, metric: str = "ip",
                      compute_dtype: DTypeLike = jnp.bfloat16) -> tuple[JitFn, JitFn]:
    kw: dict[str, Any] = {"metric": metric, "compute_dtype": compute_dtype}
    return (jax.jit(lambda e: prepare_dense(e, None, **kw)),
            jax.jit(lambda q, idx: approx_search(q, idx, k=k, recall_target=recall_target, **kw)))


def make_approx(k: int, *, recall_target: float = 0.95, metric: str = "ip",
                compute_dtype: DTypeLike = jnp.bfloat16, has_mask: bool = False) -> JitFn:
    kw: dict[str, Any] = {"metric": metric, "compute_dtype": compute_dtype}
    sk: dict[str, Any] = {"k": k, "recall_target": recall_target, **kw}
    if has_mask:
        return jax.jit(lambda q, e, m: approx_search(q, prepare_dense(e, m, **kw), **sk))
    return jax.jit(lambda q, e: approx_search(q, prepare_dense(e, None, **kw), **sk))


def approx_topk(queries: Array, db: Array, k: int, *, recall_target: float = 0.95,
                metric: str = "ip", mask: Array | None = None,
                compute_dtype: DTypeLike = jnp.bfloat16) -> tuple[Array, Array]:
    f = make_approx(k, recall_target=recall_target, metric=metric,
                    compute_dtype=compute_dtype, has_mask=mask is not None)
    out: tuple[Array, Array] = f(queries, db) if mask is None else f(queries, db, mask)
    return out
