"""V2: scan over db blocks, per-block top-k, running merge.

Never holds the full M x N score matrix live at once (one M x B block at a time). The roofline
model still counts "modeled HBM traffic": each block's scores are written and re-read.

Also stage-split helpers for R5: score only (full M x N scores) and select only (top_k on scores).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.typing import DTypeLike

from trbench._common import (
    JitFn,
    block_topk,
    dense_scores,
    init_running,
    l2_normalize,
    merge_topk,
    prepare_dense,
)

DEFAULT_BLOCK = 65536


class BlockIndex(NamedTuple):
    eb: Array  # (nb, B, D) compute_dtype, padded rows zero
    esq: Array | None  # (nb, B) f32 ||e||^2 (l2) else None
    vb: Array  # (nb, B) bool; padded and masked rows False


def prepare_blockwise(
    db: Array, mask: Array | None, *, metric: str, compute_dtype: DTypeLike, block_size: int
) -> BlockIndex:
    n = db.shape[0]
    b = min(block_size, n)
    nb = -(-n // b)
    npad = nb * b - n
    e = db.astype(jnp.float32)
    if metric == "cos":
        e = l2_normalize(e)
    valid = jnp.ones((n,), bool) if mask is None else mask.astype(bool)
    esq = jnp.sum(e * e, axis=1) if metric == "l2" else None
    if npad:
        e = jnp.pad(e, ((0, npad), (0, 0)))
        valid = jnp.pad(valid, (0, npad))
        esq = None if esq is None else jnp.pad(esq, (0, npad))
    return BlockIndex(
        e.astype(compute_dtype).reshape(nb, b, -1),
        None if esq is None else esq.reshape(nb, b),
        valid.reshape(nb, b),
    )


def search_blockwise(
    queries: Array, idx: BlockIndex, *, k: int, metric: str, compute_dtype: DTypeLike
) -> tuple[Array, Array]:
    m = queries.shape[0]
    nb, b = idx.vb.shape
    q = queries.astype(jnp.float32)
    if metric == "cos":
        q = l2_normalize(q)
    qc = q.astype(compute_dtype)
    esq = idx.esq if idx.esq is not None else jnp.zeros((nb, b), jnp.float32)

    def step(
        carry: tuple[Array, Array], xs: tuple[Array, Array, Array, Array]
    ) -> tuple[tuple[Array, Array], None]:
        run_v, run_i = carry
        e_blk, esq_blk, v_blk, off = xs
        s = jnp.matmul(qc, e_blk.T, preferred_element_type=jnp.float32)
        if metric == "l2":
            s = 2.0 * s - esq_blk[None, :]
        s = jnp.where(v_blk[None, :], s, -jnp.inf)
        bv, bi = block_topk(s, k, off)
        return merge_topk(run_v, run_i, bv, bi, k), None

    offs = jnp.arange(nb, dtype=jnp.int32) * b
    (vals, ids), _ = lax.scan(step, init_running(m, k), (idx.eb, esq, idx.vb, offs))
    return vals, ids


def make_blockwise_parts(
    k: int,
    *,
    metric: str = "ip",
    compute_dtype: DTypeLike = jnp.bfloat16,
    has_mask: bool = False,
    block_size: int = DEFAULT_BLOCK,
) -> tuple[JitFn, JitFn]:
    kw: dict[str, Any] = {"metric": metric, "compute_dtype": compute_dtype}
    pk: dict[str, Any] = {**kw, "block_size": block_size}
    if has_mask:
        prep_fn = jax.jit(lambda e, mk: prepare_blockwise(e, mk, **pk))
    else:
        prep_fn = jax.jit(lambda e: prepare_blockwise(e, None, **pk))
    return prep_fn, jax.jit(lambda q, idx: search_blockwise(q, idx, k=k, **kw))


def make_blockwise(
    k: int,
    *,
    metric: str = "ip",
    compute_dtype: DTypeLike = jnp.bfloat16,
    has_mask: bool = False,
    block_size: int = DEFAULT_BLOCK,
) -> JitFn:
    kw: dict[str, Any] = {"metric": metric, "compute_dtype": compute_dtype}
    pk: dict[str, Any] = {**kw, "block_size": block_size}
    if has_mask:
        return jax.jit(
            lambda q, e, mk: search_blockwise(q, prepare_blockwise(e, mk, **pk), k=k, **kw)
        )
    return jax.jit(lambda q, e: search_blockwise(q, prepare_blockwise(e, None, **pk), k=k, **kw))


def blockwise(
    queries: Array,
    db: Array,
    k: int,
    *,
    metric: str = "ip",
    mask: Array | None = None,
    compute_dtype: DTypeLike = jnp.bfloat16,
    block_size: int = DEFAULT_BLOCK,
) -> tuple[Array, Array]:
    f = make_blockwise(
        k,
        metric=metric,
        compute_dtype=compute_dtype,
        has_mask=mask is not None,
        block_size=block_size,
    )
    out: tuple[Array, Array] = f(queries, db) if mask is None else f(queries, db, mask)
    return out


def make_score_only_parts(
    *, metric: str = "ip", compute_dtype: DTypeLike = jnp.bfloat16
) -> tuple[JitFn, JitFn]:
    """(prepare(db) -> DenseIndex, score(queries, index) -> (M, N) f32)."""
    kw: dict[str, Any] = {"metric": metric, "compute_dtype": compute_dtype}
    return (
        jax.jit(lambda e: prepare_dense(e, None, **kw)),
        jax.jit(lambda q, idx: dense_scores(q, idx, **kw)),
    )


def make_score_only(
    *, metric: str = "ip", compute_dtype: DTypeLike = jnp.bfloat16, has_mask: bool = False
) -> JitFn:
    """Scoring stage of V1 alone: (queries, db[, mask]) -> full (M, N) f32 scores."""
    kw: dict[str, Any] = {"metric": metric, "compute_dtype": compute_dtype}
    if has_mask:
        return jax.jit(lambda q, e, m: dense_scores(q, prepare_dense(e, m, **kw), **kw))
    return jax.jit(lambda q, e: dense_scores(q, prepare_dense(e, None, **kw), **kw))


def make_select_only(k: int) -> JitFn:
    """Selection stage alone: precomputed (M, N) scores -> (vals, ids)."""

    def f(scores: Array) -> tuple[Array, Array]:
        v, i = lax.top_k(scores, k)
        return v, i.astype(jnp.int32)

    return jax.jit(f)
