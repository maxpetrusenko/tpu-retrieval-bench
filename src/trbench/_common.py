"""Shared helpers for blockwise / Pallas / approx variants."""

from __future__ import annotations

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
