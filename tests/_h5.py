import numpy as np
import pytest

from trbench import datasets as ds


def make_file(path, distance="euclidean", n=200, nq=7, d=8, seed=0):
    h5py = pytest.importorskip("h5py")  # only dataset tests need it
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
