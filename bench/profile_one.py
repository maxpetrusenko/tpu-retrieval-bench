"""Capture a jax.profiler trace of one jit_topk config. Run as a script (`!python`) so the
notebook kernel never imports jax and never holds the TPU.

Usage: python bench/profile_one.py --m 128 --n 1000000 --d 128 --k 10 --out results/profile_x
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from trbench.jit_topk import make_jit_topk


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name, default in (("m", 128), ("n", 1_000_000), ("d", 128), ("k", 10), ("seed", 0)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out = a.out or f"results/profile_jit_topk_m{a.m}_n{a.n}_d{a.d}"
    rng = np.random.default_rng(a.seed)
    q = jnp.asarray(rng.standard_normal((a.m, a.d), dtype=np.float32))
    e = jnp.asarray(rng.standard_normal((a.n, a.d), dtype=np.float32))
    f = make_jit_topk(a.k)
    jax.block_until_ready(f(q, e))  # type: ignore[no-untyped-call]  # jax is untyped here
    with jax.profiler.trace(out):
        jax.block_until_ready(f(q, e))  # type: ignore[no-untyped-call]  # jax is untyped here
    print("trace written to", out)


if __name__ == "__main__":
    main()
