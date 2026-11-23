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
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.typing import DTypeLike

from trbench._common import (
    JitFn,
    PallasRef,
    block_topk,
    init_running,
    l2_normalize,
    merge_topk,
    round_up,
    shape_struct,
)

DEFAULT_BM, DEFAULT_BN, DEFAULT_BK = 256, 512, 512
DEFAULT_CHUNK = 65536


def default_interpret() -> bool:
    return jax.default_backend() != "tpu"


def _kernel(
    q_ref: PallasRef,
    e_ref: PallasRef,
    aux_ref: PallasRef,
    o_ref: PallasRef,
    acc_ref: PallasRef,
    *,
    nk: int,
    is_l2: bool,
) -> None:
    kk = pl.program_id(2)

    @pl.when(kk == 0)
    def _() -> None:
        acc_ref[...] = jnp.zeros_like(acc_ref)

    acc_ref[...] += lax.dot_general(
        q_ref[...], e_ref[...], (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    )

    @pl.when(kk == nk - 1)
    def _() -> None:
        s = acc_ref[...]
        if is_l2:
            s = 2.0 * s - aux_ref[0:1, :]
        o_ref[...] = jnp.where(aux_ref[1:2, :] > 0.5, s, -jnp.inf)


def score_tiles(
    q: Array, e: Array, aux: Array, *, bm: int, bn: int, bk: int, is_l2: bool, interpret: bool
) -> Array:
    """q (Mp, Dp), e (Cn, Dp), aux (8, Cn) -> scores (Mp, Cn) f32. Dims are block multiples."""
    mp, dp = q.shape
    cn = e.shape[0]
    nk = dp // bk
    out: Array = pl.pallas_call(  # pallas_call is typed as returning Any
        partial(_kernel, nk=nk, is_l2=is_l2),
        grid=(mp // bm, cn // bn, nk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((bn, bk), lambda i, j, kk: (j, kk)),
            pl.BlockSpec((8, bn), lambda i, j, kk: (0, j)),
        ],
        out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
        out_shape=shape_struct((mp, cn), jnp.float32),
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(q, e, aux)
    return out


def resolve_blocks(
    m: int, n: int, d: int, bm: int, bn: int, bk: int, chunk: int
) -> tuple[int, int, int, int]:
    """Clamp blocks for small shapes; chunk is a multiple of bn. Returns (bm, bn, bk, cn)."""
    bm = min(bm, round_up(m, 16))
    bn = min(bn, round_up(n, 128))
    bk = min(bk, round_up(d, 128))
    cn = min(max(bn, (chunk // bn) * bn), round_up(n, bn))
    return bm, bn, bk, cn


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["ecs", "auxs", "e_rem", "aux_rem"],
    meta_fields=["bn", "bk", "cn"],
)
@dataclass
class PallasIndex:
    """Prepared db: padded, compute dtype, aux (norms + valid) built once.

    ecs/auxs: full chunks (nfull, cn, Dp) / (nfull, 8, cn). e_rem/aux_rem: one smaller last
    chunk padded only to bn (not to a full chunk), or None.
    """

    ecs: Array
    auxs: Array
    e_rem: Array | None
    aux_rem: Array | None
    bn: int
    bk: int
    cn: int


def prepare_pallas(
    db: Array,
    mask: Array | None,
    *,
    metric: str,
    compute_dtype: DTypeLike,
    bn: int,
    bk: int,
    chunk: int,
) -> PallasIndex:
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
    return PallasIndex(
        ecs, auxs, ec[full:] if rem else None, aux[:, full:] if rem else None, bn, bk, cn
    )


def padded_dim(idx: PallasIndex) -> int:
    """Padded feature dim Dp of a prepared index (from a full chunk, else the remainder)."""
    if idx.ecs.shape[0]:
        return int(idx.ecs.shape[2])
    assert idx.e_rem is not None, "index has no rows"
    return int(idx.e_rem.shape[1])


def search_pallas(
    queries: Array,
    idx: PallasIndex,
    *,
    k: int,
    metric: str,
    compute_dtype: DTypeLike,
    bm: int,
    interpret: bool,
) -> tuple[Array, Array]:
    m, d = queries.shape
    bm = min(bm, round_up(m, 16))
    mp = round_up(m, bm)
    dp = padded_dim(idx)
    q = queries.astype(jnp.float32)
    if metric == "cos":
        q = l2_normalize(q)
    qc = jnp.pad(q.astype(compute_dtype), ((0, mp - m), (0, dp - d)))
    kw: dict[str, Any] = {
        "bm": bm,
        "bn": idx.bn,
        "bk": idx.bk,
        "is_l2": metric == "l2",
        "interpret": interpret,
    }
    carry = init_running(mp, k)
    if idx.ecs.shape[0]:

        def step(
            carry: tuple[Array, Array], xs: tuple[Array, Array, Array]
        ) -> tuple[tuple[Array, Array], None]:
            e_c, aux_c, off = xs
            s = score_tiles(qc, e_c, aux_c, **kw)
            bv, bi = block_topk(s, k, off)
            return merge_topk(*carry, bv, bi, k), None

        offs = jnp.arange(idx.ecs.shape[0], dtype=jnp.int32) * idx.cn
        carry, _ = lax.scan(step, carry, (idx.ecs, idx.auxs, offs))
    if idx.e_rem is not None and idx.aux_rem is not None:
        s = score_tiles(qc, idx.e_rem, idx.aux_rem, **kw)
        bv, bi = block_topk(s, k, idx.ecs.shape[0] * idx.cn)
        carry = merge_topk(*carry, bv, bi, k)
    vals, ids = carry
    return vals[:m], ids[:m]


def make_pallas_parts(
    k: int,
    *,
    metric: str = "ip",
    compute_dtype: DTypeLike = jnp.bfloat16,
    has_mask: bool = False,
    bm: int = DEFAULT_BM,
    bn: int = DEFAULT_BN,
    bk: int = DEFAULT_BK,
    chunk: int = DEFAULT_CHUNK,
    interpret: bool | None = None,
) -> tuple[JitFn, JitFn]:
    interpret = default_interpret() if interpret is None else interpret
    pk: dict[str, Any] = {
        "metric": metric,
        "compute_dtype": compute_dtype,
        "bn": bn,
        "bk": bk,
        "chunk": chunk,
    }
    sk: dict[str, Any] = {
        "k": k,
        "metric": metric,
        "compute_dtype": compute_dtype,
        "bm": bm,
        "interpret": interpret,
    }
    if has_mask:
        prep_fn = jax.jit(lambda e, mk: prepare_pallas(e, mk, **pk))
    else:
        prep_fn = jax.jit(lambda e: prepare_pallas(e, None, **pk))
    return prep_fn, jax.jit(lambda q, idx: search_pallas(q, idx, **sk))


def make_pallas_score(
    k: int,
    *,
    metric: str = "ip",
    compute_dtype: DTypeLike = jnp.bfloat16,
    has_mask: bool = False,
    bm: int = DEFAULT_BM,
    bn: int = DEFAULT_BN,
    bk: int = DEFAULT_BK,
    chunk: int = DEFAULT_CHUNK,
    interpret: bool | None = None,
) -> JitFn:
    interpret = default_interpret() if interpret is None else interpret
    pk: dict[str, Any] = {
        "metric": metric,
        "compute_dtype": compute_dtype,
        "bn": bn,
        "bk": bk,
        "chunk": chunk,
    }
    sk: dict[str, Any] = {
        "k": k,
        "metric": metric,
        "compute_dtype": compute_dtype,
        "bm": bm,
        "interpret": interpret,
    }
    if has_mask:
        return jax.jit(lambda q, e, mk: search_pallas(q, prepare_pallas(e, mk, **pk), **sk))
    return jax.jit(lambda q, e: search_pallas(q, prepare_pallas(e, None, **pk), **sk))


def pallas_score(
    queries: Array,
    db: Array,
    k: int,
    *,
    metric: str = "ip",
    mask: Array | None = None,
    compute_dtype: DTypeLike = jnp.bfloat16,
    **kw: Any,
) -> tuple[Array, Array]:
    f = make_pallas_score(
        k, metric=metric, compute_dtype=compute_dtype, has_mask=mask is not None, **kw
    )
    out: tuple[Array, Array] = f(queries, db) if mask is None else f(queries, db, mask)
    return out
