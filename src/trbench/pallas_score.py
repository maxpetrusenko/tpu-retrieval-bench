"""V3: Pallas tiled matmul with fused l2 term + mask epilogue; selection in JAX by db chunks.

Aux operand (8, Np) f32 per db: row 0 = ||e||^2, row 1 = valid (1.0 / 0.0). 8 rows keeps the
block (8, bn) tile-aligned for Mosaic. The full M x N is never live at once (the roofline model
still counts block scores as modeled HBM traffic): each scan step scores one chunk of `chunk`
columns (a multiple of bn), takes its top-k, and merges. The last chunk is only padded to bn,
not to a full chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from trbench._common import block_topk, init_running, l2_normalize, merge_topk, round_up

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
    """Clamp blocks for small shapes; chunk is a multiple of bn. Returns (bm, bn, bk, cn)."""
    bm = min(bm, round_up(m, 16))
    bn = min(bn, round_up(n, 128))
    bk = min(bk, round_up(d, 128))
    cn = min(max(bn, (chunk // bn) * bn), round_up(n, bn))
    return bm, bn, bk, cn


@partial(jax.tree_util.register_dataclass,
         data_fields=["ecs", "auxs", "e_rem", "aux_rem"], meta_fields=["bn", "bk", "cn"])
@dataclass
class PallasIndex:
    """Prepared db: padded, compute dtype, aux (norms + valid) built once.

    ecs/auxs: full chunks (nfull, cn, Dp) / (nfull, 8, cn). e_rem/aux_rem: one smaller last
    chunk padded only to bn (not to a full chunk), or None.
    """

    ecs: object
    auxs: object
    e_rem: object
    aux_rem: object
    bn: int
    bk: int
    cn: int


def prepare_pallas(db, mask, *, metric, compute_dtype, bn, bk, chunk):
    n, d = db.shape
    _, bn, bk, cn = resolve_blocks(16, n, d, 16, bn, bk, chunk)
    dp, np_ = round_up(d, bk), round_up(n, bn)
    e = db.astype(jnp.float32)
    if metric == "cos":
        e = l2_normalize(e)
    esq = jnp.sum(e * e, axis=1) if metric == "l2" else jnp.zeros((n,), jnp.float32)
    valid = jnp.ones((n,), jnp.float32) if mask is None else mask.astype(jnp.float32)
    ec = jnp.pad(e.astype(compute_dtype), ((0, np_ - n), (0, dp - d)))
    aux = jnp.zeros((8, np_), jnp.float32)
    aux = aux.at[0, :n].set(esq).at[1, :n].set(valid)  # padded columns stay invalid
    nfull = np_ // cn
    full = nfull * cn
    ecs = ec[:full].reshape(nfull, cn, dp)
    auxs = aux[:, :full].reshape(8, nfull, cn).transpose(1, 0, 2)
    rem = np_ > full
    return PallasIndex(ecs, auxs, ec[full:] if rem else None, aux[:, full:] if rem else None,
                       bn, bk, cn)


def search_pallas(queries, idx, *, k, metric, compute_dtype, bm, interpret):
    m, d = queries.shape
    bm = min(bm, round_up(m, 16))
    mp, dp = round_up(m, bm), idx.ecs.shape[2] if idx.ecs.shape[0] else idx.e_rem.shape[1]
    q = queries.astype(jnp.float32)
    if metric == "cos":
        q = l2_normalize(q)
    qc = jnp.pad(q.astype(compute_dtype), ((0, mp - m), (0, dp - d)))
    kw = {"bm": bm, "bn": idx.bn, "bk": idx.bk, "is_l2": metric == "l2", "interpret": interpret}
    carry = init_running(mp, k)
    if idx.ecs.shape[0]:
        def step(carry, xs):
            e_c, aux_c, off = xs
            s = score_tiles(qc, e_c, aux_c, **kw)
            bv, bi = block_topk(s, k, off)
            return merge_topk(*carry, bv, bi, k), None

        offs = jnp.arange(idx.ecs.shape[0], dtype=jnp.int32) * idx.cn
        carry, _ = lax.scan(step, carry, (idx.ecs, idx.auxs, offs))
    if idx.e_rem is not None:
        s = score_tiles(qc, idx.e_rem, idx.aux_rem, **kw)
        bv, bi = block_topk(s, k, idx.ecs.shape[0] * idx.cn)
        carry = merge_topk(*carry, bv, bi, k)
    vals, ids = carry
    return vals[:m], ids[:m]


def make_pallas_parts(k, *, metric="ip", compute_dtype=jnp.bfloat16, has_mask=False,
                      bm=DEFAULT_BM, bn=DEFAULT_BN, bk=DEFAULT_BK, chunk=DEFAULT_CHUNK,
                      interpret=None):
    interpret = default_interpret() if interpret is None else interpret
    pk = {"metric": metric, "compute_dtype": compute_dtype, "bn": bn, "bk": bk, "chunk": chunk}
    sk = {"k": k, "metric": metric, "compute_dtype": compute_dtype, "bm": bm,
          "interpret": interpret}
    if has_mask:
        prep_fn = jax.jit(lambda e, mk: prepare_pallas(e, mk, **pk))
    else:
        prep_fn = jax.jit(lambda e: prepare_pallas(e, None, **pk))
    return prep_fn, jax.jit(lambda q, idx: search_pallas(q, idx, **sk))


def make_pallas_score(k, *, metric="ip", compute_dtype=jnp.bfloat16, has_mask=False,
                      bm=DEFAULT_BM, bn=DEFAULT_BN, bk=DEFAULT_BK, chunk=DEFAULT_CHUNK,
                      interpret=None):
    interpret = default_interpret() if interpret is None else interpret
    pk = {"metric": metric, "compute_dtype": compute_dtype, "bn": bn, "bk": bk, "chunk": chunk}
    sk = {"k": k, "metric": metric, "compute_dtype": compute_dtype, "bm": bm,
          "interpret": interpret}
    if has_mask:
        return jax.jit(lambda q, e, mk: search_pallas(q, prepare_pallas(e, mk, **pk), **sk))
    return jax.jit(lambda q, e: search_pallas(q, prepare_pallas(e, None, **pk), **sk))


def pallas_score(queries, db, k, *, metric="ip", mask=None, compute_dtype=jnp.bfloat16, **kw):
    f = make_pallas_score(k, metric=metric, compute_dtype=compute_dtype,
                          has_mask=mask is not None, **kw)
    return f(queries, db) if mask is None else f(queries, db, mask)
