"""Smoke test for bench/profile_one.py: runs the real jit_topk on CPU at a tiny shape."""

from __future__ import annotations

from pathlib import Path

import pytest
from bench import profile_one


def test_profile_one_writes_trace_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "trace"
    profile_one.main(["--m", "4", "--n", "256", "--d", "8", "--k", "3", "--out", str(out)])
    assert out.is_dir()
    assert any(out.rglob("*")), "profiler wrote nothing"
    assert f"trace written to {out}" in capsys.readouterr().out


def test_profile_one_default_out_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    profile_one.main(["--m", "2", "--n", "64", "--d", "4", "--k", "2"])
    assert "results/profile_jit_topk_m2_n64_d4" in capsys.readouterr().out
    assert (tmp_path / "results/profile_jit_topk_m2_n64_d4").is_dir()
