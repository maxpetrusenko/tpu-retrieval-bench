"""V3: Pallas tiled matmul with fused l2 term + mask epilogue; selection in JAX by db chunks.

Aux operand (8, Np) f32 per db: row 0 = ||e||^2, row 1 = valid (1.0 / 0.0). 8 rows keeps the
block (8, bn) tile-aligned for Mosaic. Full M x N is never materialized: each scan step scores
one chunk of `chunk` columns (a multiple of bn), takes its top-k, and merges.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from trbench._common import block_topk, init_running, merge_topk, prep, round_up

DEFAULT_BM, DEFAULT_BN, DEFAULT_BK = 256, 512, 512
DEFAULT_CHUNK = 65536


def default_interpret() -> bool:
    return jax.default_backend() != "tpu"


def _kernel(q_ref, e_ref, aux_ref, o_ref, acc_ref, *, nk, is_l2):
    kk = pl.program_id(2)

    @pl.when(kk == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += lax.dot_general(
        q_ref[...], e_ref[...], (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )

    @pl.when(kk == nk - 1)
    def _():
        s = acc_ref[...]
        if is_l2:
            s = 2.0 * s - aux_ref[0:1, :]
        o_ref[...] = jnp.where(aux_ref[1:2, :] > 0.5, s, -jnp.inf)


def score_tiles(q, e, aux, *, bm, bn, bk, is_l2, interpret):
    """q (Mp, Dp), e (Cn, Dp), aux (8, Cn) -> scores (Mp, Cn) f32. Dims are block multiples."""
    mp, dp = q.shape
    cn = e.shape[0]
    nk = dp // bk
    return pl.pallas_call(
        partial(_kernel, nk=nk, is_l2=is_l2),
        grid=(mp // bm, cn // bn, nk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((bn, bk), lambda i, j, kk: (j, kk)),
            pl.BlockSpec((8, bn), lambda i, j, kk: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        out_shape=jax.ShapeDtypeStruct((mp, cn), jnp.float32),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(q, e, aux)


def resolve_blocks(m, n, d, bm, bn, bk, chunk):
    """Clamp blocks for small shapes; chunk is a multiple of bn."""
    bm = min(bm, round_up(m, 16))
    bn = min(bn, round_up(n, 128))
    bk = min(bk, round_up(d, 128))
    cn = min(max(bn, (chunk // bn) * bn), round_up(n, bn))
    return bm, bn, bk, cn


def pallas_impl(queries, db, mask, *, k, metric, compute_dtype, bm, bn, bk, chunk, interpret):
    m, d = queries.shape
    n = db.shape[0]
    bm, bn, bk, cn = resolve_blocks(m, n, d, bm, bn, bk, chunk)
    mp, dp, np_ = round_up(m, bm), round_up(d, bk), round_up(n, cn)
    q, e = prep(queries, db, metric)
    esq = jnp.sum(e * e, axis=1) if metric == "l2" else jnp.zeros((n,), jnp.float32)
    valid = jnp.ones((n,), jnp.float32) if mask is None else mask.astype(jnp.float32)
    qc = jnp.pad(q.astype(compute_dtype), ((0, mp - m), (0, dp - d)))
    ec = jnp.pad(e.astype(compute_dtype), ((0, np_ - n), (0, dp - d)))
    aux = jnp.zeros((8, np_), jnp.float32)
    aux = aux.at[0, :n].set(esq).at[1, :n].set(valid)  # padded columns stay invalid
    nc = np_ // cn
    ecs = ec.reshape(nc, cn, dp)
    auxs = aux.reshape(8, nc, cn).transpose(1, 0, 2)

    def step(carry, xs):
        run_v, run_i = carry
        e_c, aux_c, off = xs
        s = score_tiles(qc, e_c, aux_c, bm=bm, bn=bn, bk=bk, is_l2=metric == "l2",
                        interpret=interpret)
        bv, bi = block_topk(s, k, off)
        return merge_topk(run_v, run_i, bv, bi, k), None

    offs = jnp.arange(nc, dtype=jnp.int32) * cn
    (vals, ids), _ = lax.scan(step, init_running(mp, k), (ecs, auxs, offs))
    return vals[:m], ids[:m]


def make_pallas_score(k, *, metric="ip", compute_dtype=jnp.bfloat16, has_mask=False,
                      bm=DEFAULT_BM, bn=DEFAULT_BN, bk=DEFAULT_BK, chunk=DEFAULT_CHUNK,
                      interpret=None):
    interpret = default_interpret() if interpret is None else interpret
    kw = {"k": k, "metric": metric, "compute_dtype": compute_dtype, "bm": bm, "bn": bn,
          "bk": bk, "chunk": chunk, "interpret": interpret}
    if has_mask:
        return jax.jit(lambda q, e, mk: pallas_impl(q, e, mk, **kw))
    return jax.jit(lambda q, e: pallas_impl(q, e, None, **kw))


def pallas_score(queries, db, k, *, metric="ip", mask=None, compute_dtype=jnp.bfloat16, **kw):
    f = make_pallas_score(k, metric=metric, compute_dtype=compute_dtype,
                          has_mask=mask is not None, **kw)
    return f(queries, db) if mask is None else f(queries, db, mask)
