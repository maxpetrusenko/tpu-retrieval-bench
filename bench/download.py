"""Download ann-benchmarks HDF5 files to data/ with a byte-size check (never run in CI)."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from trbench.datasets import DATA_DIR, DATASETS

URL = "https://ann-benchmarks.com/{}.hdf5"
# ann-benchmarks.com answered HTTP 403 to the default Python-urllib agent (Colab, 2026-11-18).
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
ALIASES = {"sift": "sift-128-euclidean", "glove": "glove-100-angular", "gist": "gist-960-euclidean"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream an https url to dest with a browser-like User-Agent. Other schemes are refused."""
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError(f"refusing non-https download url: {url!r}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with (
        dest.open("wb") as f,
        urllib.request.urlopen(req) as resp,  # nosec B310 - https only, checked above
    ):
        shutil.copyfileobj(resp, f, 1 << 20)


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
        download(URL.format(name), tmp)
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
