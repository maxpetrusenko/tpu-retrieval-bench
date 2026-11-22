"""V0: float64 NumPy oracle. Ties: lower id wins. Invalid tail slots: id -1, score -inf."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

METRICS = ("ip", "l2", "cos")


def _normalize(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-30)


def scores_ref(queries: npt.ArrayLike, db: npt.ArrayLike, *, metric: str = "ip",
               mask: npt.ArrayLike | None = None) -> npt.NDArray[np.float64]:
    """Full (M, N) float64 score matrix; masked candidates are -inf."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}")
    q = np.asarray(queries, dtype=np.float64)
    e = np.asarray(db, dtype=np.float64)
    if metric == "cos":
        q, e = _normalize(q), _normalize(e)
    s = q @ e.T
    if metric == "l2":
        s = 2.0 * s - np.sum(e * e, axis=1)[None, :]
    if mask is not None:
        s = np.where(np.asarray(mask, dtype=bool)[None, :], s, -np.inf)
    return s


def numpy_ref(queries: npt.ArrayLike, db: npt.ArrayLike, k: int, *, metric: str = "ip",
              mask: npt.ArrayLike | None = None,
              compute_dtype: object = None) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    """Return (scores (M,k) float32, ids (M,k) int32). compute_dtype is ignored (float64)."""
    s = scores_ref(queries, db, metric=metric, mask=mask)
    # stable sort on negated scores keeps lower id first among ties
    ids = np.argsort(-s, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(s, ids, axis=1)
    ids = np.where(np.isneginf(top), -1, ids)  # invalid slots: id -1, score -inf
    return top.astype(np.float32), ids.astype(np.int32)
