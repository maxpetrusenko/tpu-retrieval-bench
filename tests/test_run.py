import json

import pytest
from bench import run

h5py = pytest.importorskip("h5py")
from tests.test_datasets import make_file


def read_all(root):
    return {p.parent.name: json.loads(p.read_text()) for p in root.rglob("*.json")}


def test_stage_split_nulls_and_ratio(tmp_path):
    run.main(["--algos", "score_only,select_only,jit_topk", "--m", "4", "--n", "2048", "--d", "16",
              "--k", "5", "--out", str(tmp_path)])
    recs = read_all(tmp_path)
    sel, sc, jt = recs["select_only"]["metrics"], recs["score_only"]["metrics"], recs["jit_topk"]
    assert sel["flops_achieved"] is None and sel["percent_of_peak"] is None
    assert sel["selection_time_ms"] > 0 and sel["selection_over_matmul"] > 0
    assert sc["selection_over_matmul"] == sel["selection_over_matmul"]
    assert sc["flops_achieved"] > 0
    for r in recs.values():
        assert {"roofline_label", "bytes_moved", "arithmetic_intensity"} <= set(r["metrics"])
    assert jt["metrics"]["bytes_moved"] > 0


@pytest.mark.parametrize("dist", ["euclidean", "angular"])
def test_dataset_flag(tmp_path, dist):
    make_file(tmp_path / "sift-128-euclidean.hdf5", dist, n=300, nq=6, d=16)
    run.main(["--dataset", "sift-128-euclidean", "--data-dir", str(tmp_path), "--algos",
              "jit_topk,blockwise", "--k", "10", "--dtype", "f32", "--n-train", "250",
              "--n-test", "5", "--out", str(tmp_path / "out")])
    recs = read_all(tmp_path / "out")
    assert set(recs) == {"jit_topk", "blockwise"}
    for r in recs.values():
        assert r["dataset"]["name"] == "sift-128-euclidean"
        assert (r["dataset"]["n_train"], r["dataset"]["n_test"]) == (250, 5)
        assert r["metrics"]["recall"] == 1.0
    assert "sift-128-euclidean" in str(next((tmp_path / "out").rglob("*.json")))
