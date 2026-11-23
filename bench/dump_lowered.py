"""Lowered StableHLO for jit_topk and pallas_score aimed at TPU, produced on any host (CPU ok).

This is the LOWERED program (before XLA optimization and TPU compilation), not compiled HLO.
It shows which ops the program asks for (top_k, dot_general, tpu_custom_call), not what the TPU
compiler fused or scheduled. For compiled HLO run bench/run.py --dump-hlo on the TPU.
Usage: python -m bench.dump_lowered --m 128 --n 1000000 --d 128 --k 10 --out results/tpu/lowered
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from trbench.jit_topk import make_jit_topk_parts
from trbench.pallas_score import make_pallas_parts


def lowered_text(fn: Any, *args: Any) -> str:
    return str(fn.trace(*args).lower(lowering_platforms=("tpu",)).as_text())


def dump(m: int, n: int, d: int, k: int, out: str | Path) -> dict[str, Path]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # jax.ShapeDtypeStruct is untyped in the pinned jax
    q = jax.ShapeDtypeStruct((m, d), jnp.float32)  # type: ignore[no-untyped-call]  # untyped in the pinned jax
    e = jax.ShapeDtypeStruct((n, d), jnp.float32)  # type: ignore[no-untyped-call]  # untyped in the pinned jax
    paths: dict[str, Path] = {}
    prep, search = make_jit_topk_parts(k)
    idx = jax.eval_shape(prep, e)
    paths["jit_topk"] = out / f"jit_topk_m{m}_n{n}_d{d}_k{k}.stablehlo.txt"
    paths["jit_topk"].write_text(lowered_text(search, q, idx))
    prep, search = make_pallas_parts(k, interpret=False)
    idx = jax.eval_shape(prep, e)
    paths["pallas_score"] = out / f"pallas_score_m{m}_n{n}_d{d}_k{k}.stablehlo.txt"
    paths["pallas_score"].write_text(lowered_text(search, q, idx))
    return paths


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (("m", 128), ("n", 1_000_000), ("d", 128), ("k", 10)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--out", default="results/tpu/lowered")
    a = p.parse_args(argv)
    for name, path in dump(a.m, a.n, a.d, a.k, a.out).items():
        print(name, path, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
