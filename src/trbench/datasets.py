"""ann-benchmarks HDF5 loader and recall metrics (distance-threshold k-nn recall)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

METRIC_MAP = {"euclidean": "l2", "angular": "cos"}
DATASETS = {
    "sift-128-euclidean": 525_128_288,
    "glove-100-angular": 485_413_888,
    "gist-960-euclidean": 3_844_648_288,
}
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def resolve(name_or_path) -> Path:
    p = Path(name_or_path)
    if p.suffix == ".hdf5" or p.exists():
        return p
    return DATA_DIR / f"{name_or_path}.hdf5"


def bruteforce_gt(train, test, metric, k=100, chunk=1024):
    """Exact (neighbors, distances) in ann-benchmarks convention; used for n_train subsets."""
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    if metric == "cos":
        train = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-30)
        test = test / np.maximum(np.linalg.norm(test, axis=1, keepdims=True), 1e-30)
    k = min(k, len(train))
    e2 = np.sum(train.astype(np.float64) ** 2, axis=1)
    ids = np.empty((len(test), k), dtype=np.int64)
    dists = np.empty((len(test), k), dtype=np.float64)
    for s in range(0, len(test), chunk):
        q = test[s : s + chunk]
        dot = (q @ train.T).astype(np.float64)
        if metric == "l2":
            d = np.sum(q.astype(np.float64) ** 2, axis=1, keepdims=True) - 2 * dot + e2[None, :]
            d = np.sqrt(np.maximum(d, 0.0))
        else:
            d = 1.0 - dot
        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        pd = np.take_along_axis(d, part, axis=1)
        order = np.argsort(pd, axis=1, kind="stable")
        ids[s : s + chunk] = np.take_along_axis(part, order, axis=1)
        dists[s : s + chunk] = np.take_along_axis(pd, order, axis=1)
    return ids, dists


def load_ann(name_or_path, n_train=None, n_test=None) -> dict:
    """Load an ann-benchmarks HDF5 -> dict(train, test, neighbors, distances, metric).

    metric: euclidean -> "l2", angular -> "cos". If n_train truncates the corpus the stored
    ground truth is invalid, so it is recomputed by brute force on the subset.
    """
    import h5py

    path = resolve(name_or_path)
    with h5py.File(path, "r") as f:
        raw = f.attrs.get("distance", "euclidean")
        raw = raw.decode() if isinstance(raw, bytes) else str(raw)
        if raw not in METRIC_MAP:
            raise ValueError(f"unsupported distance attr {raw!r} in {path}")
        metric = METRIC_MAP[raw]
        full_n = f["train"].shape[0]
        train = np.asarray(f["train"][: n_train or full_n], dtype=np.float32)
        test = np.asarray(f["test"][:n_test], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"][:n_test])
        distances = np.asarray(f["distances"][:n_test])
    if len(train) < full_n:
        neighbors, distances = bruteforce_gt(train, test, metric)
    return {
        "train": train, "test": test, "neighbors": neighbors,
        "distances": distances, "metric": metric,
    }


def scores_to_distances(scores, metric, queries=None):
    """Convert search scores (higher better) to ann-benchmarks distances (lower better)."""
    s = np.asarray(scores, dtype=np.float64)
    if metric == "ip":
        return -s
    if metric == "cos":
        return 1.0 - s
    if metric == "l2":
        q2 = np.sum(np.asarray(queries, dtype=np.float64) ** 2, axis=1, keepdims=True)
        return np.sqrt(np.maximum(q2 - s, 0.0))
    raise ValueError(f"unknown metric {metric!r}")


def knn_recall(found_dists, true_dists, k, eps=1e-3) -> float:
    """ann-benchmarks k-nn recall: a found item counts if its distance <= kth true + eps."""
    found = np.asarray(found_dists, dtype=np.float64)[:, :k]
    thresh = np.asarray(true_dists, dtype=np.float64)[:, k - 1 : k] + eps
    return float(np.mean(np.sum(found <= thresh, axis=1) / k))


def id_set_recall(found_ids, true_ids, k) -> float:
    """Mean |found[:k] intersect true[:k]| / k. Stricter than knn_recall under ties."""
    f, t = np.asarray(found_ids)[:, :k], np.asarray(true_ids)[:, :k]
    hits = [len(set(a.tolist()) & set(b.tolist())) for a, b in zip(f, t, strict=True)]
    return float(np.mean(hits) / k)
