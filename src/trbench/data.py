"""Seeded synthetic data."""

from __future__ import annotations

import hashlib

import numpy as np


def synthetic(m, n, d, *, seed=0, mask_frac=0.0):
    """Return (queries f32 [m,d], db f32 [n,d], mask bool [n] | None)."""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((m, d), dtype=np.float32)
    e = rng.standard_normal((n, d), dtype=np.float32)
    mask = None
    if mask_frac > 0:
        mask = rng.random(n) >= mask_frac
    return q, e, mask


def data_hash(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        if a is not None:
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def recall_at_k(ref_scores, got_scores, eps=1e-3) -> float:
    """ann-benchmarks distance-threshold recall: got counts if score >= kth ref score - tol."""
    ref = np.asarray(ref_scores, dtype=np.float64)
    got = np.asarray(got_scores, dtype=np.float64)
    thresh = ref[:, -1:] - eps * np.maximum(np.abs(ref[:, -1:]), 1.0)
    return float(np.mean(np.sum(got >= thresh, axis=1) / ref.shape[1]))
