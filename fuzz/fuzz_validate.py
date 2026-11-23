"""Coverage-guided fuzz harness (atheris) for the result validator and record parsing.

    pip install atheris
    python fuzz/fuzz_validate.py -max_total_time=60 fuzz/corpus

The target feeds raw bytes to `trbench.validate.validate_json_bytes`, which must return a list
of problems for any input and never raise. An uncaught exception is a finding. The corpus
directory is optional and may start empty. The Hypothesis tests in `tests/test_fuzz.py` check
the same contract inside the normal suite (and `tests/test_fuzz.py` loads this file against a
stub `atheris`, so the target itself is exercised without an atheris build).
"""

import sys

import atheris

with atheris.instrument_imports():
    from trbench.validate import validate_json_bytes


def test_one_input(data: bytes) -> None:
    errs = validate_json_bytes(data)
    if not isinstance(errs, list):
        raise TypeError("validate_json_bytes must return a list")


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
