"""V4: Pallas fused matmul + per-tile partial top-k. The (bm, bn) score tile never reaches HBM.

Two stages (Samaga 2026 style two-stage top-k):
  1. Kernel: same tiling, accumulator and epilogue (l2 term, mask) as pallas_score. On the last
     k-step the tile is reduced in VMEM to a partial top-k' and only (bm, k') values and global
     ids per (i, j) tile are written, shape (M, n_tiles * k') per db chunk.
  2. JAX: lax.top_k over the M x (n_tiles * k') candidates, merged across chunks.

`pallas_fused_topk` (exact when kprime == k): selection is k' rounds of row max, argmax via an
iota compare (lowest column wins ties), then mask out the pick. Any global top-k element is in
its own tile's top-k, so stage 2 over the per-tile top-k is exact. Ties: ids stay ascending among
equal values (tile order, then selection order), so the lower-id-wins policy of the other
variants holds. Masked/padded columns are -inf; they are taken last, lowest id first. Padded
ids (>= N) become -1.

Cost model. Matmul tile: 2*bm*bn*D FLOP on the MXU. Selection: k' rounds, each ~3 passes over
the bm x bn tile (max, argmin, mask) on the VPU, so ~3*k'*bm*bn VPU ops per tile versus
2*bm*bn*D MXU FLOP. VPU peak is far below MXU peak, so selection overtakes the matmul once
k' * 3 / (2*D) * (MXU/VPU throughput ratio) > 1. Wins: small k (<= ~16) and large N, where it
removes the 2 * 4 * M * N score write+read; loses: k near 128 (selection ~ k' x the cost of the
matmul epilogue; use pallas_score + lax.top_k) and small M (nothing to hide). HBM extra:
M * n_tiles * k' * 8 bytes of candidates (see roofline.bytes_moved). The kernel writes lane-
padded blocks (k' rounded up to 128), stage 2 slices them to k'.

`pallas_fused_bucket` (approximate, exact=False): TPU-KNN / approx_max_k style. Each tile column
c falls in bin c mod L (L lanes, a multiple of 128). Per bin keep the running max over the
bn / L slabs (one compare + select per element, O(bm*bn) VPU, independent of k). A true top-k
element is lost only when another top-k element of the same row lands in the same bin of the
same tile, so recall falls as k grows relative to n_tiles * L. Recall is measured, not assumed.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.typing import DTypeLike

from trbench._common import JitFn, PallasRef, l2_normalize, merge_topk, round_up, shape_struct
from trbench.pallas_score import (
    DEFAULT_BK,
    DEFAULT_BM,
    DEFAULT_BN,
    DEFAULT_CHUNK,
    PallasIndex,
    default_interpret,
    padded_dim,
    prepare_pallas,
)

MAX_K = 128
LANES = 128
NEG = float(np.finfo(np.float32).min)  # stand-in for -inf until a slot is taken


def _accumulate(q_ref: PallasRef, e_ref: PallasRef, aux_ref: PallasRef, acc_ref: PallasRef, *,
                nk: int, is_l2: bool) -> None:
    """Shared matmul step; returns nothing. On the last k-step acc_ref holds masked scores."""
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
        s = jnp.where(aux_ref[1:2, :] > 0.5, s, -jnp.inf)
        acc_ref[...] = jnp.where(s == -jnp.inf, NEG, s)  # -inf is reserved for "taken"


def _topk_kernel(q_ref: PallasRef, e_ref: PallasRef, aux_ref: PallasRef, ov_ref: PallasRef,
                 oi_ref: PallasRef, acc_ref: PallasRef, *, nk: int, is_l2: bool,
                 kprime: int) -> None:
    j = pl.program_id(1)
    kk = pl.program_id(2)
    _accumulate(q_ref, e_ref, aux_ref, acc_ref, nk=nk, is_l2=is_l2)

    @pl.when(kk == nk - 1)
    def _() -> None:
        bm, bn = acc_ref.shape
        kp = ov_ref.shape[1]
        col = lax.broadcasted_iota(jnp.int32, (bm, bn), 1)
        lane = lax.broadcasted_iota(jnp.int32, (bm, kp), 1)

        def body(t: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            ov, oi = carry
            w = acc_ref[...]
            mx = jnp.max(w, axis=1, keepdims=True)
            pos = jnp.min(jnp.where(w == mx, col, bn), axis=1, keepdims=True)
            acc_ref[...] = jnp.where(col == pos, -jnp.inf, w)
            val = jnp.where(mx <= NEG, -jnp.inf, mx)
            sel = lane == t
            return jnp.where(sel, val, ov), jnp.where(sel, j * bn + pos, oi)

        init = (jnp.full((bm, kp), -jnp.inf, jnp.float32), jnp.full((bm, kp), -1, jnp.int32))
        ov, oi = lax.fori_loop(0, kprime, body, init)
        ov_ref[...] = ov
        oi_ref[...] = oi


def _bucket_kernel(q_ref: PallasRef, e_ref: PallasRef, aux_ref: PallasRef, ov_ref: PallasRef,
                   oi_ref: PallasRef, acc_ref: PallasRef, *, nk: int, is_l2: bool,
                   lanes: int) -> None:
    j = pl.program_id(1)
    kk = pl.program_id(2)
    _accumulate(q_ref, e_ref, aux_ref, acc_ref, nk=nk, is_l2=is_l2)

    @pl.when(kk == nk - 1)
    def _() -> None:
        bm, bn = acc_ref.shape
        best = acc_ref[:, 0:lanes]
        slab_id = jnp.zeros((bm, lanes), jnp.int32)
        for s in range(1, bn // lanes):  # strict > keeps the lower column on ties
            cur = acc_ref[:, s * lanes:(s + 1) * lanes]
            better = cur > best
            best = jnp.where(better, cur, best)
            slab_id = jnp.where(better, s, slab_id)
        lane = lax.broadcasted_iota(jnp.int32, (bm, lanes), 1)
        ov_ref[...] = jnp.where(best <= NEG, -jnp.inf, best)
        oi_ref[...] = j * bn + slab_id * lanes + lane


def fused_tiles(q: Array, e: Array, aux: Array, *, bm: int, bn: int, bk: int, is_l2: bool,
                interpret: bool, mode: str, kprime: int,
                lanes: int) -> tuple[tuple[Array, Array], int]:
    """q (Mp, Dp), e (Cn, Dp), aux (8, Cn) -> candidate values/ids (Mp, n_tiles * width).

    width = k' rounded up to 128 (topk) or `lanes` (bucket). Dims are block multiples.
    """
    mp, dp = q.shape
    cn = e.shape[0]
    nk, nt = dp // bk, cn // bn
    if mode == "topk":
        width = round_up(kprime, LANES)
        kern = partial(_topk_kernel, nk=nk, is_l2=is_l2, kprime=kprime)
    else:
        width = lanes
        kern = partial(_bucket_kernel, nk=nk, is_l2=is_l2, lanes=lanes)
    out_spec = pl.BlockSpec((bm, width), lambda i, j, kk: (i, j))
    cand: tuple[Array, Array] = pl.pallas_call(  # pallas_call is typed as returning Any
        kern,
        grid=(mp // bm, nt, nk),
        in_specs=[
            pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
            pl.BlockSpec((bn, bk), lambda i, j, kk: (j, kk)),
            pl.BlockSpec((8, bn), lambda i, j, kk: (0, j)),
        ],
        out_specs=[out_spec, out_spec],
        out_shape=[
            shape_struct((mp, nt * width), jnp.float32),
            shape_struct((mp, nt * width), jnp.int32),
        ],
        scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=interpret,
    )(q, e, aux)
    return cand, width


def _stage2(cv: Array, ci: Array, *, nt: int, width: int, keep: int, k: int,
            offset: Array | int) -> tuple[Array, Array]:
    """lax.top_k over per-tile candidates (first `keep` of each `width` slots) -> (M, k)."""
    mp = cv.shape[0]
    if keep < width:
        cv = cv.reshape(mp, nt, width)[:, :, :keep].reshape(mp, nt * keep)
        ci = ci.reshape(mp, nt, width)[:, :, :keep].reshape(mp, nt * keep)
    kb = min(k, cv.shape[1])
    v, pos = lax.top_k(cv, kb)
    ids = jnp.take_along_axis(ci, pos, axis=1)
    ids = jnp.where(ids >= 0, ids + offset, -1)
    if kb < k:
        v = jnp.pad(v, ((0, 0), (0, k - kb)), constant_values=-jnp.inf)
        ids = jnp.pad(ids, ((0, 0), (0, k - kb)), constant_values=-1)
    return v, ids


def search_fused(queries: Array, idx: PallasIndex, *, k: int, metric: str,
                 compute_dtype: DTypeLike, bm: int, interpret: bool, mode: str, kprime: int,
                 lanes: int, n: int) -> tuple[Array, Array]:
    m, d = queries.shape
    bm = min(bm, round_up(m, 16))
    mp = round_up(m, bm)
    dp = padded_dim(idx)
    q = queries.astype(jnp.float32)
    if metric == "cos":
        q = l2_normalize(q)
    qc = jnp.pad(q.astype(compute_dtype), ((0, mp - m), (0, dp - d)))
    kw: dict[str, Any] = {"bm": bm, "bn": idx.bn, "bk": idx.bk, "is_l2": metric == "l2", "interpret": interpret,
          "mode": mode, "kprime": kprime, "lanes": lanes}
    keep = kprime if mode == "topk" else lanes

    def chunk(e_c: Array, aux_c: Array, off: Array | int) -> tuple[Array, Array]:
        (cv, ci), width = fused_tiles(qc, e_c, aux_c, **kw)
        return _stage2(cv, ci, nt=e_c.shape[0] // idx.bn, width=width, keep=keep, k=k,
                       offset=off)

    # No -inf/-1 init carry: it would win -inf ties against real (masked) ids. Seed with the
    # first chunk instead.
    nfull = idx.ecs.shape[0]
    offs = jnp.arange(nfull, dtype=jnp.int32) * idx.cn
    if nfull:
        carry = chunk(idx.ecs[0], idx.auxs[0], offs[0])
        if nfull > 1:
            def step(carry: tuple[Array, Array],
                     xs: tuple[Array, Array, Array]) -> tuple[tuple[Array, Array], None]:
                e_c, aux_c, off = xs
                return merge_topk(*carry, *chunk(e_c, aux_c, off), k), None

            carry, _ = lax.scan(step, carry, (idx.ecs[1:], idx.auxs[1:], offs[1:]))
        if idx.e_rem is not None and idx.aux_rem is not None:
            carry = merge_topk(*carry, *chunk(idx.e_rem, idx.aux_rem, nfull * idx.cn), k)
    else:
        assert idx.e_rem is not None and idx.aux_rem is not None, "index has no rows"
        carry = chunk(idx.e_rem, idx.aux_rem, 0)
    vals, ids = carry
    # padded columns and masked (-inf) slots are never real results: id -1 like every variant
    ids = jnp.where((ids >= n) | jnp.isneginf(vals), -1, ids)
    return vals[:m], ids[:m]


def check_args(k: int, mode: str, kprime: int, lanes: int, bn: int) -> None:
    if mode not in ("topk", "bucket"):
        raise ValueError(f"unknown mode {mode!r}")
    if not 1 <= k <= MAX_K:
        raise ValueError(f"fused kernels support 1 <= k <= {MAX_K}, got {k}")
    if mode == "topk" and not 1 <= kprime <= k:
        raise ValueError(f"kprime must be in [1, k={k}], got {kprime}")
    if mode == "bucket" and (lanes % LANES or bn % lanes):
        raise ValueError(f"lanes must be a multiple of {LANES} dividing bn={bn}, got {lanes}")


def make_fused_parts(k: int, n: int, *, metric: str = "ip",
                     compute_dtype: DTypeLike = jnp.bfloat16, has_mask: bool = False,
                     bm: int = DEFAULT_BM, bn: int = DEFAULT_BN, bk: int = DEFAULT_BK,
                     chunk: int = DEFAULT_CHUNK, kprime: int | None = None, mode: str = "topk",
                     lanes: int = LANES,
                     interpret: bool | None = None) -> tuple[JitFn, JitFn]:
    """(prepare, search) for a db of n rows (static). prepare(db[, mask]) is untimed.

    kprime defaults to k (exact); kprime < k is approximate.
    """
    interpret = default_interpret() if interpret is None else interpret
    kprime = k if kprime is None else kprime
    check_args(k, mode, kprime, lanes, bn)
    pk: dict[str, Any] = {"metric": metric, "compute_dtype": compute_dtype, "bn": bn, "bk": bk, "chunk": chunk}
    if has_mask:
        prep_fn = jax.jit(lambda e, mk: prepare_pallas(e, mk, **pk))
    else:
        prep_fn = jax.jit(lambda e: prepare_pallas(e, None, **pk))
    sk: dict[str, Any] = {"k": k, "metric": metric, "compute_dtype": compute_dtype, "bm": bm,
          "interpret": interpret, "mode": mode, "kprime": kprime, "lanes": lanes, "n": n}
    return prep_fn, jax.jit(lambda q, idx: search_fused(q, idx, **sk))


def make_pallas_fused(k: int, n: int, **kw: Any) -> JitFn:
    """Jitted one-shot (queries, db[, mask]) -> (vals, ids)."""
    prep, search = make_fused_parts(k, n, **kw)
    if kw.get("has_mask"):
        return jax.jit(lambda q, e, mk: search(q, prep(e, mk)))
    return jax.jit(lambda q, e: search(q, prep(e)))


def pallas_fused_topk(queries: Array, db: Array, k: int, *, metric: str = "ip",
                      mask: Array | None = None, compute_dtype: DTypeLike = jnp.bfloat16,
                      **kw: Any) -> tuple[Array, Array]:
    f = make_pallas_fused(k, db.shape[0], metric=metric, compute_dtype=compute_dtype,
                          has_mask=mask is not None, **kw)
    out: tuple[Array, Array] = f(queries, db) if mask is None else f(queries, db, mask)
    return out


def pallas_fused_bucket(queries: Array, db: Array, k: int, **kw: Any) -> tuple[Array, Array]:
    return pallas_fused_topk(queries, db, k, mode="bucket", **kw)
