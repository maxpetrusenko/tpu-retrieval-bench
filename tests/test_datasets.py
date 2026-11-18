import numpy as np
import pytest

from trbench import datasets as ds

h5py = pytest.importorskip("h5py")


def make_file(path, distance="euclidean", n=200, nq=7, d=8, seed=0):
    rng = np.random.default_rng(seed)
    train = rng.standard_normal((n, d), dtype=np.float32)
    test = rng.standard_normal((nq, d), dtype=np.float32)
    metric = ds.METRIC_MAP[distance]
    nb, dist = ds.bruteforce_gt(train, test, metric, k=100)
    with h5py.File(path, "w") as f:
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=test)
        f.create_dataset("neighbors", data=nb)
        f.create_dataset("distances", data=dist)
        f.attrs["distance"] = distance
    return train, test


@pytest.mark.parametrize("dist,metric", [("euclidean", "l2"), ("angular", "cos")])
def test_load_and_metric_map(tmp_path, dist, metric):
    p = tmp_path / "x.hdf5"
    make_file(p, dist)
    r = ds.load_ann(p)
    assert r["metric"] == metric
    assert r["train"].shape == (200, 8) and r["neighbors"].shape == (7, 100)
    r2 = ds.load_ann(p, n_test=3)
    assert r2["test"].shape[0] == 3 and r2["neighbors"].shape[0] == 3


def test_n_train_recomputes_ground_truth(tmp_path):
    p = tmp_path / "x.hdf5"
    train, test = make_file(p)
    r = ds.load_ann(p, n_train=50)
    assert r["train"].shape[0] == 50
    assert r["neighbors"].max() < 50
    d = np.linalg.norm(test[0][None] - train[:50], axis=1)
    np.testing.assert_array_equal(r["neighbors"][0][:10], np.argsort(d)[:10])


def test_recall_functions(tmp_path):
    p = tmp_path / "x.hdf5"
    make_file(p)
    r = ds.load_ann(p)
    k = 10
    assert ds.knn_recall(r["distances"], r["distances"], k) == 1.0
    assert ds.id_set_recall(r["neighbors"], r["neighbors"], k) == 1.0
    # shift ids by one rank: 9/10 overlap in the id sense
    shifted = r["neighbors"][:, 1 : k + 1]
    assert ds.id_set_recall(shifted, r["neighbors"], k) == pytest.approx(0.9)
    # distances far worse than threshold -> 0
    assert ds.knn_recall(r["distances"] + 10.0, r["distances"], k) == 0.0
    # tie within eps counts
    assert ds.knn_recall(r["distances"][:, :k] + 5e-4, r["distances"], k) >= 0.9


def test_scores_to_distances_l2_matches_numpy():
    rng = np.random.default_rng(1)
    q, e = rng.standard_normal((4, 6)), rng.standard_normal((9, 6))
    score = 2 * q @ e.T - np.sum(e * e, axis=1)
    d = ds.scores_to_distances(score, "l2", q)
    np.testing.assert_allclose(d, np.linalg.norm(q[:, None] - e[None], axis=2), atol=1e-6)


def test_resolve_name():
    assert ds.resolve("sift-128-euclidean").name == "sift-128-euclidean.hdf5"
