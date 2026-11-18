"""Download ann-benchmarks HDF5 files to data/ with a byte-size check (never run in CI)."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from trbench.datasets import DATA_DIR, DATASETS

URL = "http://ann-benchmarks.com/{}.hdf5"
ALIASES = {"sift": "sift-128-euclidean", "glove": "glove-100-angular", "gist": "gist-960-euclidean"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, out: Path) -> Path:
    name = ALIASES.get(name, name)
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name}; choose from {sorted(DATASETS)}")
    dest = out / f"{name}.hdf5"
    expected = DATASETS[name]
    if dest.exists() and dest.stat().st_size == expected:
        print(f"skip {dest} (present, {expected} bytes)")
    else:
        out.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        print(f"downloading {URL.format(name)} -> {dest}")
        urllib.request.urlretrieve(URL.format(name), tmp)
        got = tmp.stat().st_size
        if got != expected:
            tmp.unlink()
            raise SystemExit(f"size mismatch for {name}: got {got}, expected {expected}")
        tmp.rename(dest)
    print(f"sha256 {name} {sha256(dest)}")
    return dest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("names", nargs="+", help="sift | glove | gist or full dataset name")
    p.add_argument("--out", default=str(DATA_DIR))
    a = p.parse_args(argv)
    for n in a.names:
        fetch(n, Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
