"""Algorithm registry for bench.run: name -> factory returning (prepare, search, ...).

Each factory returns (prepare, search, record name, exact, params). prepare(db) -> index is
untimed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from trbench import roofline
from trbench._common import JitFn
from trbench.approx import make_approx_parts
from trbench.blockwise import make_blockwise_parts, make_score_only_parts, make_select_only
from trbench.jit_topk import make_jit_topk_parts
from trbench.pallas_fused import make_fused_parts
from trbench.pallas_score import make_pallas_parts

if TYPE_CHECKING:  # annotation only; bench.cli imports this module for DTYPES
    from bench.cli import Args

BlockParams = tuple[int, int, int]
Built = tuple[JitFn, JitFn, str, bool, dict[str, Any]]
Factory = Callable[..., Built]

DTYPES = {"bf16": jnp.bfloat16, "f32": jnp.float32}
MATERIALIZING = roofline.MATERIALIZING | {"score_only", "select_only"}
EXACT_ALGOS = ("jit_topk", "blockwise", "pallas_score", "pallas_fused_topk")
FUSED = ("pallas_fused_topk", "pallas_fused_bucket")
PALLAS_ALGOS = ("pallas_score", *FUSED)
STAGE_ONLY = ("score_only", "select_only")


def _jit(k: int, dtype: Any, a: Args, bp: BlockParams | None) -> Built:
    p, s = make_jit_topk_parts(k, metric=a.metric, compute_dtype=dtype)
    return p, s, "jit_matmul_topk", True, {}


def _blockwise(k: int, dtype: Any, a: Args, bp: BlockParams | None) -> Built:
    p, s = make_blockwise_parts(k, metric=a.metric, compute_dtype=dtype,
                                block_size=a.block_size)
    return p, s, "blockwise_topk", True, {"block_size": a.block_size}


def _pallas(k: int, dtype: Any, a: Args, bp: BlockParams | None) -> Built:
    assert bp is not None, "pallas variants always run with block params"
    bm, bn, bk = bp
    p, s = make_pallas_parts(k, metric=a.metric, compute_dtype=dtype, bm=bm, bn=bn, bk=bk,
                             chunk=a.chunk)
    return p, s, "pallas_score_topk", True, {"bm": bm, "bn": bn, "bk": bk, "chunk": a.chunk,
                                             "interpret": jax.default_backend() != "tpu"}


def _fused(mode: str) -> Factory:
    def factory(k: int, dtype: Any, a: Args, bp: BlockParams | None,
                n: int | None = None) -> Built:
        assert bp is not None and n is not None, "fused variants need block params and db size"
        bm, bn, bk = bp
        kprime = a.kprime or k
        p, s = make_fused_parts(k, n, metric=a.metric, compute_dtype=dtype, bm=bm, bn=bn, bk=bk,
                                chunk=a.chunk, kprime=kprime, mode=mode, lanes=a.lanes)
        params = {"bm": bm, "bn": bn, "bk": bk, "chunk": a.chunk,
                  "interpret": jax.default_backend() != "tpu"}
        params.update({"kprime": kprime} if mode == "topk" else {"lanes": a.lanes})
        return p, s, f"pallas_fused_{mode}", mode == "topk" and kprime == k, params
    return factory


def _approx(k: int, dtype: Any, a: Args, bp: BlockParams | None) -> Built:
    p, s = make_approx_parts(k, recall_target=a.recall_target, metric=a.metric,
                             compute_dtype=dtype)
    return p, s, "approx_max_k", False, {"recall_target": a.recall_target}


def _score_only(k: int, dtype: Any, a: Args, bp: BlockParams | None) -> Built:
    p, s = make_score_only_parts(metric=a.metric, compute_dtype=dtype)
    return p, s, "score_only", True, {}


def _select_only(k: int, dtype: Any, a: Args, bp: BlockParams | None) -> Built:
    p, s = make_score_only_parts(metric=a.metric, compute_dtype=dtype)
    return p, make_select_only(k), "select_only", True, {"score_fn": s}


FACTORIES: dict[str, Factory] = {
    "jit_topk": _jit, "blockwise": _blockwise, "pallas_score": _pallas,
    "pallas_fused_topk": _fused("topk"), "pallas_fused_bucket": _fused("bucket"),
    "approx_max_k": _approx, "score_only": _score_only, "select_only": _select_only}


def build(algo: str, k: int, dtype: Any, a: Args, bp: BlockParams | None, n: int) -> Built:
    """Factory call; fused factories need the db size (static in the jitted search)."""
    if algo in FUSED:
        return FACTORIES[algo](k, dtype, a, bp, n)
    return FACTORIES[algo](k, dtype, a, bp)
