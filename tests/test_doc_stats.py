import json
from pathlib import Path

import pytest
from bench import doc_stats

ROOT = Path(__file__).resolve().parents[1]


def _rec(dirty):
    return json.dumps({"metrics": {}, "git": {"commit": "a", "dirty": dirty}})


def _tree(tmp_path):
    for rel, dirty in [
        ("a/x.json", True),
        ("a/y/z.json", False),
        ("b/w.json", False),
        ("top.json", True),
    ]:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_rec(dirty))
    (tmp_path / "b" / "notes.json").write_text(json.dumps({"error": "oom"}))
    (tmp_path / "b" / "bad.json").write_text("{nope")
    return tmp_path


def test_collect_counts(tmp_path):
    s = doc_stats.collect(_tree(tmp_path))
    assert s["records"] == 4 and s["dirty"] == 2 and s["other_json"] == 2
    assert s["groups"] == {
        "a": {"records": 2, "dirty": 1},
        "b": {"records": 1, "dirty": 0},
        ".": {"records": 1, "dirty": 1},
    }


def test_render_and_splice_roundtrip(tmp_path):
    block = doc_stats.render(doc_stats.collect(_tree(tmp_path)), "results/x")
    assert "true on 2 of the 4 records under `results/x/` (2 clean)" in block
    assert "| `a` | 2 | 1 | 1 |" in block
    doc = f"intro\n{doc_stats.BEGIN}\nstale\n{doc_stats.END}\noutro\n"
    new = doc_stats.splice(doc, block)
    assert new.startswith("intro\n") and new.endswith("\noutro\n") and "stale" not in new
    assert doc_stats.splice(new, block) == new  # idempotent
    with pytest.raises(ValueError):
        doc_stats.splice("no markers", block)


def test_cli_check_and_write(tmp_path, capsys):
    root = _tree(tmp_path / "res")
    doc = tmp_path / "doc.md"
    doc.write_text(f"x\n{doc_stats.BEGIN}\nold\n{doc_stats.END}\n")
    assert doc_stats.main(["--results", str(root), "--check", str(doc)]) == 1
    assert "stale" in capsys.readouterr().err
    assert doc_stats.main(["--results", str(root), "--write", str(doc)]) == 0
    assert doc_stats.main(["--results", str(root), "--check", str(doc)]) == 0
    (root / "a" / "x.json").write_text(_rec(False))  # JSON changes, doc does not
    assert doc_stats.main(["--results", str(root), "--check", str(doc)]) == 1


def test_results_md_dirty_counts_match_json(monkeypatch):
    """docs/RESULTS.md quotes record and dirty counts only inside the generated block."""
    if not (ROOT / doc_stats.DEFAULT_ROOT).exists():
        pytest.skip("no committed results/tpu")
    monkeypatch.chdir(ROOT)  # the block names the root as results/tpu
    assert doc_stats.main(["--check", "docs/RESULTS.md"]) == 0
