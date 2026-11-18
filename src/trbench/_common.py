"""Shared helpers for blockwise / Pallas / approx variants."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import lax


def l2_normalize(x):
    n = jnp.linalg.norm(x, axis=1, keepdims=True)
    return x / jnp.maximum(n, 1e-30)


def prep(queries, db, metric):
    """f32 views; cos normalizes rows."""
    q = queries.astype(jnp.float32)
    e = db.astype(jnp.float32)
    if metric == "cos":
        q, e = l2_normalize(q), l2_normalize(e)
    return q, e


def round_up(x, m):
    return -(-x // m) * m


def merge_topk(run_v, run_i, blk_v, blk_i, k):
    """Merge running top-k with a block's top-k. Running entries come first (lower ids win ties)."""
    v = jnp.concatenate([run_v, blk_v], axis=1)
    i = jnp.concatenate([run_i, blk_i], axis=1)
    top_v, pos = lax.top_k(v, k)
    return top_v, jnp.take_along_axis(i, pos, axis=1)


def init_running(m, k):
    return jnp.full((m, k), -jnp.inf, jnp.float32), jnp.full((m, k), -1, jnp.int32)


def block_topk(s, k, offset):
    """top-k of a (M, W) score block with global ids (offset + local)."""
    kb = min(k, s.shape[1])
    v, i = lax.top_k(s, kb)
    i = i.astype(jnp.int32) + offset
    if kb < k:  # pad so merge shapes stay static
        pad = k - kb
        v = jnp.pad(v, ((0, 0), (0, pad)), constant_values=-jnp.inf)
        i = jnp.pad(i, ((0, 0), (0, pad)), constant_values=-1)
    return v, i


class DenseIndex(NamedTuple):
    """Prepared db for V1/A1/score_only: stored in compute dtype, aux terms precomputed."""

    ec: object  # (N, D) compute_dtype, L2-normalized first for cos
    esq: object  # (N,) f32 ||e||^2 (l2 only) else None
    valid: object  # (N,) bool or None


def prepare_dense(db, mask, *, metric, compute_dtype):
    """One-time index build (outside the timed region)."""
    e = db.astype(jnp.float32)
    if metric == "cos":
        e = l2_normalize(e)
    esq = jnp.sum(e * e, axis=1) if metric == "l2" else None
    valid = None if mask is None else mask.astype(bool)
    return DenseIndex(e.astype(compute_dtype), esq, valid)


def dense_scores(queries, idx, *, metric, compute_dtype):
    """Full (M, N) f32 scores from a prepared DenseIndex."""
    q = queries.astype(jnp.float32)
    if metric == "cos":
        q = l2_normalize(q)
    s = jnp.matmul(q.astype(compute_dtype), idx.ec.T, preferred_element_type=jnp.float32)
    if metric == "l2":
        s = 2.0 * s - idx.esq[None, :]
    if idx.valid is not None:
        s = jnp.where(idx.valid[None, :], s, -jnp.inf)
    return s
