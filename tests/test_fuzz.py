"""Property and fuzz tests for the result validator and record parsing (Hypothesis).

Contract under test: whatever bytes, JSON value or mutated record come in, `validate_record`
and `validate_json_bytes` return a list of problem strings and never raise. Select with
`pytest -m fuzz`; CI runs them with more examples (`FUZZ_EXAMPLES=2000`). The same entry
point is driven by the coverage-guided atheris harness in `fuzz/fuzz_validate.py`.
"""

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from trbench.validate import validate_json_bytes, validate_record, validate_tree

pytestmark = pytest.mark.fuzz

ROOT = Path(__file__).resolve().parents[1]
SEED_RECORDS = sorted((ROOT / "results" / "tpu" / "run4").rglob("*.json"))[:40]
EXAMPLES = int(os.environ.get("FUZZ_EXAMPLES", "300"))
FUZZ = settings(
    max_examples=EXAMPLES,
    deadline=None,
    derandomize="FUZZ_EXAMPLES" not in os.environ,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=20),
    st.sampled_from(
        [10**400, -(10**400), 1e308, -1e308, 5e-324, 0, -0.0, float("nan"), float("inf")]
    ),
)
json_values = st.recursive(
    scalars,
    lambda kids: st.one_of(
        st.lists(kids, max_size=5), st.dictionaries(st.text(max_size=12), kids, max_size=6)
    ),
    max_leaves=30,
)


def _problems_ok(errs: object) -> bool:
    return isinstance(errs, list) and all(isinstance(e, str) for e in errs)


@FUZZ
@given(st.binary(max_size=600))
def test_arbitrary_bytes_never_crash_the_parser(data):
    assert _problems_ok(validate_json_bytes(data))


@FUZZ
@given(json_values)
def test_arbitrary_json_values_never_crash_the_validator(value):
    errs = validate_json_bytes(json.dumps(value))
    assert _problems_ok(errs)
    if not isinstance(value, dict):
        assert errs  # only an object can be a record


@FUZZ
@given(st.text(max_size=300))
def test_arbitrary_text_never_crashes_the_parser(text):
    assert _problems_ok(validate_json_bytes(text))


def _load_seeds() -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in SEED_RECORDS]


def _paths(node: Any, prefix: tuple = ()) -> list[tuple]:
    out = [prefix]
    if isinstance(node, dict):
        for k, v in node.items():
            out += _paths(v, (*prefix, k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += _paths(v, (*prefix, i))
    return out


def _mutate(rec: Any, path: tuple, how: str, value: Any) -> Any:
    if not path:
        return value
    node = rec
    for step in path[:-1]:
        node = node[step]
    last = path[-1]
    if how == "delete":
        if isinstance(node, dict):
            del node[last]
        else:
            node.pop(last)
    else:
        node[last] = value
    return rec


def test_seed_records_exist_and_validate_clean():
    assert SEED_RECORDS, "results/tpu/run4 must hold seed records"
    assert all(validate_record(r) == [] for r in _load_seeds())


@FUZZ
@given(st.data())
def test_mutating_a_valid_record_returns_errors_not_exceptions(data):
    rec = copy.deepcopy(data.draw(st.sampled_from(_load_seeds())))
    for _ in range(data.draw(st.integers(1, 3))):
        path = data.draw(st.sampled_from(_paths(rec)))
        how = data.draw(st.sampled_from(["replace", "delete"]))
        rec = _mutate(rec, path, how, data.draw(json_values))
        if not isinstance(rec, dict):
            break
    assert _problems_ok(validate_json_bytes(json.dumps(rec)))


@FUZZ
@given(st.data())
def test_byte_flips_in_a_valid_record_never_crash(data):
    raw = bytearray(SEED_RECORDS[data.draw(st.integers(0, len(SEED_RECORDS) - 1))].read_bytes())
    for _ in range(data.draw(st.integers(1, 8))):
        raw[data.draw(st.integers(0, len(raw) - 1))] = data.draw(st.integers(0, 255))
    assert _problems_ok(validate_json_bytes(bytes(raw)))


def test_every_leaf_set_to_hostile_numbers_returns_problems():
    """Systematic sweep: each leaf of a valid record replaced by values that overflow floats."""
    seed = _load_seeds()[0]
    hostile = [10**400, -(10**400), 1e308, -1e308, 5e-324, float("nan"), float("inf"), True]
    for path in _paths(seed)[1:]:
        for value in hostile:
            rec = _mutate(copy.deepcopy(seed), path, "replace", value)
            assert _problems_ok(validate_record(rec)), (path, value)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\xff\xfe",
        b"{",
        b"[]",
        b"null",
        b"0",
        b'"x"',
        b"NaN",
        b"{}",
        b"[" * 100000,
        b"{" * 100000,
        b'{"a":' * 50000,
        b"1" * 10000,
        json.dumps({"metrics": {"qps": 10**400}}).encode(),
    ],
)
def test_known_hostile_payloads_return_problems(payload):
    errs = validate_json_bytes(payload)
    assert _problems_ok(errs) and errs


def test_validate_tree_reports_unreadable_files_instead_of_raising(tmp_path):
    (tmp_path / "bad.json").write_bytes(b"\xff\xfe\x00")
    (tmp_path / "deep.json").write_bytes(b"[" * 100000)
    bad = validate_tree(tmp_path)
    assert set(bad) == {str(tmp_path / "bad.json"), str(tmp_path / "deep.json")}


def test_atheris_harness_target_runs_against_a_stub(monkeypatch):
    """atheris needs a libFuzzer-enabled clang to build, so CI cannot install it. Load the
    harness against a stub module to prove the target function works and never raises."""
    import contextlib
    import importlib.util
    import types

    stub = types.ModuleType("atheris")
    stub.instrument_imports = lambda: contextlib.nullcontext()  # type: ignore[attr-defined]
    stub.Setup = lambda argv, fn: None  # type: ignore[attr-defined]
    stub.Fuzz = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "atheris", stub)
    spec = importlib.util.spec_from_file_location(
        "fuzz_validate", ROOT / "fuzz" / "fuzz_validate.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for payload in (b"", b"\xff", b"{}", SEED_RECORDS[0].read_bytes(), b"[" * 5000):
        mod.test_one_input(payload)
    mod.main()
    assert "import atheris" in (ROOT / "fuzz" / "fuzz_validate.py").read_text()
