"""Seeded synthetic data."""

from __future__ import annotations

import hashlib

import numpy as np
import numpy.typing as npt

F32 = npt.NDArray[np.float32]


def synthetic(m: int, n: int, d: int, *, seed: int = 0,
              mask_frac: float = 0.0) -> tuple[F32, F32, npt.NDArray[np.bool_] | None]:
    """Return (queries f32 [m,d], db f32 [n,d], mask bool [n] | None)."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((m, d), dtype=np.float32)
    e = rng.standard_normal((n, d), dtype=np.float32)
    mask = None
    if mask_frac > 0:
        mask = rng.random(n) >= mask_frac
    return q, e, mask


def _int_rows(rng: np.random.Generator, n: int, d: int, hi: int) -> F32:
    """Integer-valued f32 rows: matmul/norm arithmetic is exact, so ties are genuine."""
    return rng.integers(-hi, hi + 1, size=(n, d)).astype(np.float32)


def tie_heavy(m: int, n: int, d: int, *, seed: int = 0, mode: str = "duplicate",
              n_unique: int = 7,
              mask_frac: float = 0.0) -> tuple[F32, F32, npt.NDArray[np.bool_] | None]:
    """Many exactly tied scores. Integer-valued data keeps ties exact under f32.

    duplicate: db rows drawn (with repeats, shuffled) from `n_unique` distinct rows.
    quantized: db entries in {-1, 0, 1}; queries in {-1, 0, 1}, so few distinct scores.
    """
    rng = np.random.default_rng(seed)
    if mode == "duplicate":
        base = _int_rows(rng, min(n_unique, n), d, 3)
        e = base[rng.integers(0, base.shape[0], size=n)]
        q = _int_rows(rng, m, d, 3)
    elif mode == "quantized":
        e = _int_rows(rng, n, d, 1)
        q = _int_rows(rng, m, d, 1)
    else:
        raise ValueError(f"unknown tie mode {mode!r}")
    mask = rng.random(n) >= mask_frac if mask_frac > 0 else None
    return q, e, mask


def worst_case(m: int, n: int, d: int, *, kind: str = "all_equal",
               seed: int = 0) -> tuple[F32, F32, None]:
    """Adversarial selection inputs (ip/l2-exact in f32 for moderate n).

    all_equal: every db row identical, so every score ties for every query.
    ascending: score strictly increases with id (best candidates sit at the highest ids).
    descending: score strictly decreases with id (best candidates sit at the lowest ids).
    """
    rng = np.random.default_rng(seed)
    base = rng.integers(1, 4, size=(1, d)).astype(np.float32)
    q = np.repeat(base, m, axis=0)
    if kind == "all_equal":
        e = np.repeat(base, n, axis=0)
    elif kind in ("ascending", "descending"):
        c = np.arange(1, n + 1, dtype=np.float32)
        c = c if kind == "ascending" else c[::-1]
        e = c[:, None] * base
    else:
        raise ValueError(f"unknown worst-case kind {kind!r}")
    return q, np.ascontiguousarray(e), None


def data_hash(*arrays: npt.ArrayLike | None) -> str:
    h = hashlib.sha256()
    for a in arrays:
        if a is not None:
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def recall_at_k(ref_scores: npt.ArrayLike, got_scores: npt.ArrayLike,
                eps: float = 1e-3) -> float:
    """ann-benchmarks distance-threshold recall: got counts if score >= kth ref score - tol."""
    ref = np.asarray(ref_scores, dtype=np.float64)
    got = np.asarray(got_scores, dtype=np.float64)
    thresh = ref[:, -1:] - eps * np.maximum(np.abs(ref[:, -1:]), 1.0)
    return float(np.mean(np.sum(got >= thresh, axis=1) / ref.shape[1]))
