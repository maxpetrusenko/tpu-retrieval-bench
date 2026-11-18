"""Three-ceiling roofline helpers (TPU-KNN model): MXU FLOP/s, HBM bytes/s, VPU/COP.

Pure functions, no jax import. Peaks are per TensorCore from the JAX Pallas TPU hardware
reference (docs/presearch/jax-pallas-tpu.md section 4); v5e and v6e have one TC per chip.
Only chips verified there are keyed; everything else returns None (label "unknown").
"""

from __future__ import annotations

from typing import NamedTuple


class Peak(NamedTuple):
    bf16_flops: float  # FLOP/s
    hbm_bytes_per_s: float  # B/s


# keyed by jax Device.device_kind
PEAKS: dict[str, Peak] = {
    "TPU v5 lite": Peak(197e12, 820e9),
    "TPU v6 lite": Peak(920e12, 1640e9),
}

VARIANTS = (
    "jit_topk", "blockwise", "pallas_score", "pallas_fused", "approx_max_k",
    "score_only", "select_only",
)
# variants that write then re-read the full M x N f32 score matrix through HBM. blockwise and
# pallas_score write all M*N scores to HBM (in chunks) and read them back for top_k. Only a
# future pallas_fused (V4, per-tile top-k in VMEM) avoids it.
MATERIALIZING = frozenset({"jit_topk", "approx_max_k", "blockwise", "pallas_score"})
LABELS = ("MXU-bound", "HBM-bound", "VPU/COP-bound", "unknown")
_ITEMSIZE = {"bf16": 2, "bfloat16": 2, "f16": 2, "float16": 2, "f32": 4, "float32": 4}


def peak_for(device_kind: str) -> Peak | None:
    return PEAKS.get(device_kind)


def flops(m: int, n: int, d: int) -> float:
    """Scoring FLOPs: 2*M*N*D."""
    return 2.0 * m * n * d


def _itemsize(dtype) -> int:
    key = getattr(dtype, "name", None) or str(dtype)
    if key not in _ITEMSIZE:
        raise ValueError(f"unsupported dtype {dtype!r}")
    return _ITEMSIZE[key]


def bytes_moved(variant: str, m: int, n: int, d: int, k: int, dtype="bf16", *,
                q_dtype=None, db_dtype=None) -> float:
    """Modeled HBM bytes per search call.

    Reads Q (M*D) in `q_dtype` and E (N*D) in `db_dtype` (the dtype the index is stored in;
    both default to `dtype`), writes top-k (M*k f32 scores + int32 ids).
    jit_topk, approx_max_k, blockwise and pallas_score additionally write and re-read the
    M x N f32 score matrix (2 * 4 * M * N). Only pallas_fused avoids it.
    Stage helpers: score_only reads Q, E and writes the scores (no top-k, no re-read);
    select_only reads the scores and writes the top-k (no Q, E).
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {VARIANTS}")
    topk = m * k * 8
    qb = _itemsize(q_dtype or dtype)
    eb = _itemsize(db_dtype or dtype)
    reads = m * d * qb + n * d * eb
    if variant == "score_only":
        return float(reads + 4 * m * n)
    if variant == "select_only":
        return float(4 * m * n + topk)
    total = reads + topk
    if variant in MATERIALIZING:
        total += 2 * 4 * m * n
    return float(total)


def arithmetic_intensity(variant, m, n, d, k, dtype="bf16", **dt) -> float:
    """FLOP per HBM byte (select_only does no matmul: 0.0)."""
    if variant == "select_only":
        return 0.0
    return flops(m, n, d) / bytes_moved(variant, m, n, d, k, dtype, **dt)


def ridge_point(device_kind: str) -> float | None:
    """Peak FLOP/s over peak B/s (v5e ~240 FLOP/B)."""
    p = peak_for(device_kind)
    return p.bf16_flops / p.hbm_bytes_per_s if p else None


def label(
    variant: str,
    m: int,
    n: int,
    d: int,
    k: int,
    dtype,
    device_kind: str,
    *,
    select_time_s: float | None = None,
    matmul_time_s: float | None = None,
    **dt,
) -> str:
    """MXU-bound | HBM-bound | VPU/COP-bound | unknown.

    If measured selection time exceeds matmul time -> VPU/COP-bound (TPU-KNN: exact
    k-selection is bounded by the vector/COP ceiling). score_only is never labelled from the
    ratio: it has no selection. Otherwise compare ideal MXU time
    (flops/peak) with ideal HBM time (bytes/bw); the larger ceiling wins.
    """
    p = peak_for(device_kind)
    if p is None:
        return "unknown"
    if variant == "select_only":  # no matmul; only the measured stage ratio can classify it
        if select_time_s is not None and matmul_time_s is not None:
            return "VPU/COP-bound" if select_time_s > matmul_time_s else "HBM-bound"
        return "unknown"
    if (variant != "score_only" and select_time_s is not None and matmul_time_s is not None
            and select_time_s > matmul_time_s):
        return "VPU/COP-bound"
    t_mxu = flops(m, n, d) / p.bf16_flops
    t_hbm = bytes_moved(variant, m, n, d, k, dtype, **dt) / p.hbm_bytes_per_s
    return "HBM-bound" if t_hbm > t_mxu else "MXU-bound"


def percent_of_peak(achieved_flops: float, device_kind: str, dtype="bf16") -> float | None:
    """% of the bf16 peak; None for unknown chips and for f32 compute (no f32 peak in table)."""
    p = peak_for(device_kind)
    if p is None or _itemsize(dtype) != 2:
        return None
    return 100.0 * achieved_flops / p.bf16_flops
