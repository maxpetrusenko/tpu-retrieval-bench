import pytest

from trbench import roofline as rf

V5E, V6E = "TPU v5 lite", "TPU v6 lite"


def test_peaks():
    assert rf.peak_for(V5E) == (197e12, 820e9)
    assert rf.peak_for(V6E) == (920e12, 1640e9)
    assert rf.peak_for("cpu") is None
    assert rf.ridge_point(V5E) == pytest.approx(240.2, abs=0.5)


def test_flops_and_bytes():
    assert rf.flops(2, 3, 4) == 48
    fused_io = (16 * 128 + 1000 * 128) * 2 + 16 * 10 * 8
    unfused = rf.bytes_moved("jit_topk", 16, 1000, 128, 10, "bf16")
    assert unfused - fused_io == 2 * 4 * 16 * 1000
    score = 2 * 4 * 16 * 1000
    assert rf.bytes_moved("blockwise", 16, 1000, 128, 10, "bf16") == fused_io + score
    assert rf.bytes_moved("pallas_score", 16, 1000, 128, 10) == fused_io + score
    assert rf.bytes_moved("pallas_fused", 16, 1000, 128, 10, "bf16") == fused_io
    assert rf.bytes_moved("score_only", 16, 1000, 128, 10, "bf16") == (
        (16 * 128 + 1000 * 128) * 2 + 4 * 16 * 1000
    )
    assert rf.bytes_moved("select_only", 16, 1000, 128, 10) == 4 * 16 * 1000 + 16 * 10 * 8
    assert rf.bytes_moved("pallas_fused", 1, 1, 1, 1, "f32") == (1 + 1) * 4 + 8


def test_bad_inputs():
    with pytest.raises(ValueError):
        rf.bytes_moved("nope", 1, 1, 1, 1)
    with pytest.raises(ValueError):
        rf.bytes_moved("jit_topk", 1, 1, 1, 1, "int4")


def test_intensity_materialization_lowers_it():
    a = rf.arithmetic_intensity("jit_topk", 1024, 1_000_000, 128, 10)
    b = rf.arithmetic_intensity("pallas_fused", 1024, 1_000_000, 128, 10)
    assert a < b
    assert rf.arithmetic_intensity("blockwise", 1024, 1_000_000, 128, 10) == pytest.approx(a)
    assert rf.arithmetic_intensity("select_only", 1024, 1_000_000, 128, 10) == 0.0
    assert a < 40  # score write+read bounds intensity near D/4 FLOP/B


def test_labels():
    big = (1024, 1_000_000, 128, 10)
    assert rf.label("jit_topk", *big, "bf16", V5E) == "HBM-bound"
    assert rf.label("blockwise", *big, "bf16", V5E) == "HBM-bound"
    assert rf.label("pallas_fused", *big, "bf16", V5E) == "MXU-bound"
    assert rf.label("pallas_fused", 1, 1_000_000, 128, 10, "bf16", V5E) == "HBM-bound"
    assert rf.label("select_only", *big, "bf16", V5E) == "unknown"
    assert rf.label("select_only", *big, "bf16", V5E, select_time_s=2.0, matmul_time_s=1.0) == (
        "VPU/COP-bound"
    )
    assert rf.label("pallas_fused", *big, "bf16", V5E, select_time_s=2.0, matmul_time_s=1.0) == (
        "VPU/COP-bound"
    )
    assert rf.label("pallas_fused", *big, "bf16", V5E, select_time_s=0.5, matmul_time_s=1.0) == (
        "MXU-bound"
    )
    assert rf.label("jit_topk", *big, "bf16", "cpu") == "unknown"


def test_percent_of_peak():
    assert rf.percent_of_peak(98.5e12, V5E) == pytest.approx(50.0)
    assert rf.percent_of_peak(1e12, "cpu") is None


def test_db_read_uses_stored_dtype():
    base = rf.bytes_moved("jit_topk", 16, 1000, 128, 10, "bf16")
    f32_db = rf.bytes_moved("jit_topk", 16, 1000, 128, 10, "bf16", db_dtype="f32")
    assert f32_db - base == 1000 * 128 * 2
    q32 = rf.bytes_moved("jit_topk", 16, 1000, 128, 10, "bf16", q_dtype="f32")
    assert q32 - base == 16 * 128 * 2


def test_percent_of_peak_null_for_f32_compute():
    assert rf.percent_of_peak(98.5e12, V5E, "f32") is None


def test_score_only_never_labelled_from_ratio():
    big = (1024, 1_000_000, 128, 10)
    assert rf.label("score_only", *big, "bf16", V5E, select_time_s=9.0, matmul_time_s=1.0) != (
        "VPU/COP-bound"
    )
