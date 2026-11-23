# Changelog

All notable changes. Format: Keep a Changelog; versions follow `pyproject.toml`.

## Unreleased

### Added
- Run 4 (#20): the `v0.2.0` notebook rerun unmodified in a second Colab TPU v5e-1 session (`results/tpu/run4`, `run4-devtime`: 121 records, 0 dirty, strict validation passes). `bench/session_variance.py` joins two runs per cell and reports between-session variance (small shapes median 3.8%, max 35.2%; large shapes median 0.3%, max 20.2%) and which claims against `jit_topk` hold in both sessions; `results/analysis/` holds the output and the first `scripts/tpu_smoke.sh` TPU log. `docs/RESULTS.md` section 10, README, `docs/BENCHMARK_METHOD.md` rule 9 and `docs/QUALITY.md` updated; CI validates run4 with `--strict`.
- Release integrity (#21): `release.yml` is split into a read-only `build` job and a `publish` job (the only one with `contents: write`, `id-token: write`, `attestations: write`). `publish` attaches a SLSA build provenance attestation for the sdist and wheel (`actions/attest-build-provenance`), `SHA256SUMS` and a CycloneDX 1.5 SBOM (`sbom.cdx.json`, from `uv.lock`) to the GitHub release. README "Verify a release" gives `gh attestation verify` and `sha256sum -c`. `scripts/release_assets.sh` and `scripts/release_dry_run.sh` run the same logic locally (build, checksums, SBOM, CHANGELOG gate, notes) and `tests/test_release_dry_run.py` runs them.
- Fuzzing (#21): Hypothesis property and fuzz tests of the result validator and record parsing (`tests/test_fuzz.py`, marker `fuzz`, a `fuzz (hypothesis)` CI job with 2000 examples per test), a coverage-guided atheris harness (`fuzz/fuzz_validate.py`) and `trbench.validate.validate_json_bytes`, the never-raises entry point for untrusted record text.
- SAST (#21): a `sast (bandit)` CI job (`bandit -ll`, config in `pyproject.toml`). `SECURITY.md` gains a solo-maintainer review note.
- `scripts/changelog.py`: `structure` (normal test suite: headings and link references), `release` (tag-time gate) and `cut` (move Unreleased into the new version section, fix links, set the `CITATION.cff` date). `scripts/regen_docs_img.sh` regenerates the tracked `docs/img` plots from Run 3 as an explicit maintainer step.

### Changed
- The test that forced `Unreleased` to be empty is replaced: the suite checks CHANGELOG structure only, and the empty-Unreleased rule moves to the tag-time gate in `release.yml`.
- `notebooks/colab_tpu.ipynb` writes plots to `<RUN_ID>/plots` (untracked) and zips only `results`, so the plot cell no longer dirties tracked files in `docs/img`.
- Version 0.2.1 in `pyproject.toml`, `uv.lock` and `CITATION.cff`.

### Fixed
- `bench.validate_results` and `trbench.validate.validate_record` raised `OverflowError` on an integer too large for a float (for example `10**400` in a metric), instead of reporting a validation error. Found by the new fuzz tests. Undecodable or deeply nested record files are reported as unreadable instead of crashing the tree walk.
- `bench/download.py` fetched datasets over plain HTTP and passed any URL scheme to `urlopen` (bandit B310). It now uses `https://ann-benchmarks.com/` and refuses non-https URLs.

## [0.2.0] - 2026-11-22

Covers PRs #9 to #12 and the quality pass that follows them.

### Changed
- Schema 0.2 headline QPS: `metrics.qps` is now the median-pass QPS (`qps` = `qps_median`), so it sits inside `qps_ci95`, which bootstraps the same statistic. The fastest pass moves to the new required field `qps_best` (M / min time); `best_search_time_s` is unchanged. `qps_wall` and `qps_loop` follow the median too. Before, the headline was the min-time pass while the CI described the median pass, so the headline could fall outside its own CI. Schema 0.1 records (all committed TPU results) keep their best-pass headline and stay valid. No 0.2 record had been published, so 0.2 was redefined rather than bumped.
- `bench.validate_results` checks a 0.2 record for headline = median, headline inside CI, `qps_best` = M / min time and `qps_median` against `search_time_s_all`.
- `bench.summarize` and `bench.plot` compare on the median pass only when every record in a table or plot group is 0.2; a mix of 0.1 and 0.2 is compared on `qps_best`, labelled (`QPS = best pass`, speedups tagged "mixed basis", plot axes name the basis). Documented in `docs/BENCHMARK_METHOD.md` and `docs/RESULTS.md`.
- `docs/QUALITY.md`: the self-graded "met" scores and the stale "Status after this PR" table are removed; the criteria and evidence links stay, labelled a maintainer checklist, with a review log (baseline 7, post-#18 review 8.0).
- `bench/run.py` (497 lines) is split into `bench/cli.py` (arguments), `bench/algos.py` (algorithm registry), `bench/plan.py` (jobs and file names), `bench/execute.py` (timing and recall) and `bench/emit.py` (record building and writing); `bench/run.py` keeps the driver. CLI flags, file names and record contents are unchanged apart from `qps_best` and the QPS basis above. `python bench/run.py` and `python -m bench.run` both work.

### Fixed
- `notebooks/colab_tpu.ipynb` wrote its main run to `results/colab_run`, not the canonical results directory, and cloned the default branch. It now sets `REF` (default tag `v0.2.0`, bump on release) and `RUN_ID` (default `results/tpu/run-<UTC stamp>`), checks out `REF`, installs with `pip -c notebooks/constraints.txt` plus `jax[tpu]` at the locked version, and runs `PYTHON=python3 bash scripts/reproduce_all.sh {RUN_ID}`. The "Run-2 grid" wording is gone. Run 3 itself came from a hand-added cell (see `docs/RESULTS.md` section 9.11).
- `README.md` no longer says `bench/run.py` has no mask option (the arguments live in `bench/cli.py`), and the pre-split layout references in `README.md`, `docs/SPEC.md` and `docs/BENCHMARK_METHOD.md` are updated.
- `docs/RESULTS.md` preamble: Run 3 (section 9) supersedes Run 2 (section 8); the 8.7 and 8.8 caveats are marked superseded except where noted; the plots are from Run 3.
- `scripts/reproduce_all.sh` no longer assumes `python` is on PATH: it uses `$PYTHON`, else `uv run python` (uv installed and a `pyproject.toml` present), else `python3`, else `python`.

### Added
- Reproducibility: `notebooks/constraints.txt` (exported from `uv.lock` by `scripts/export_constraints.sh`, checked in sync by `tests/test_constraints.py` and a CI step); `tests/test_notebook.py` asserts the notebook run cell, pinned ref and constraints install. `scripts/tpu_smoke.sh` (one tiny shape per algorithm on a TPU, validate, compare with Run 3 through `bench/smoke_compare.py`) and the manual `tpu-smoke` workflow (`[self-hosted, tpu]`, `workflow_dispatch` only).
- Methodology: between-session variance (up to 36% at small shapes, not covered by the within-run bootstrap CI) is stated in `docs/BENCHMARK_METHOD.md`, the `docs/RESULTS.md` limitations and the README; small-shape claims are marked low-confidence.
- CI: hosted job validates `results/tpu/run3` and `run3-devtime` with `--strict`, runs `doc_stats --check`, the `reproduce_all.sh` dry run and the constraints check; pytest-cov with `fail_under = 88` (measured 90.4%); a `package` job (`uv build`, install the wheel into a fresh venv, import, `py.typed`); an `audit` job (`pip-audit` on the exported lock). The self-hosted `test` job runs on push to main and manual dispatch only, never for pull requests; the runner's isolation is documented in `SECURITY.md`.
- Security: CodeQL (python) and OpenSSF Scorecard workflows, SHA-pinned, minimal permissions; `SECURITY.md` gains a supported-versions table, a disclosure timeline and CVE handling.
- Packaging: `release.yml` (tag `v*`: build sdist and wheel, require tag = pyproject version, GitHub release with the CHANGELOG section and the dists attached; `scripts/release_notes.py`). README states the package is not on PyPI and shows install from git. `CITATION.cff` carries a TODO for a Zenodo DOI after the first release. Compare and release link references at the bottom.
- Type checking: `uv run mypy` runs strict on `src/trbench` and with `check_untyped_defs` on `bench/` (the split run modules are fully annotated). `src/trbench/py.typed` ships in the wheel. CI runs mypy in the hosted lint-test job.
- `CITATION.cff`, a README "Cite" section, `docs/QUALITY.md` (eight-dimension scorecard with pass criteria and evidence).
- CI (#11): hosted lint/test matrix on Python 3.12 and 3.13, SHA-pinned actions, dependabot, `SECURITY.md`.
- Statistics: `--reps` defaults to 20 (minimum 5 enforced); records gain stdev and bootstrap 95% CIs of p50 latency and QPS (`schema_version` 0.2, fields required from 0.2, optional for 0.1 records). `bench.summarize` flags speedups over `jit_topk` as "within noise" when QPS CIs overlap.
- Dirty-tree guard: `bench/run.py` refuses a dirty tree unless `--allow-dirty`; `git.dirty` ignores the output directory. `bench.validate_results --strict` fails dirty records, and the default run prints a dirty-record count.
- `bench/doc_stats.py` and a test keep the record and dirty counts in `docs/RESULTS.md` equal to the JSON.
- `scripts/reproduce_all.sh`: the full Run-2 grid into a fresh directory, then validate and summarize. The Colab notebook calls it as its single entry point.
- Run 2 TPU v5e-1 results (`results/tpu/run2`, `results/tpu/run2-devtime`, `docs/RESULTS.md` section 8): fused kernels, block sweeps, `approx_max_k` recall targets, loop and device timing. Plots regenerated from Run 2.
- V4 `pallas_fused_topk` (`src/trbench/pallas_fused.py`): Pallas matmul with in-kernel per-tile top-k (k <= 128, `kprime` option), stage 2 `lax.top_k`; exact. Optional `pallas_fused_bucket` (approximate, TPU-KNN style bins). Registered in `bench/run.py`, CPU interpret and TPU-lowering tests, roofline candidate bytes (k' padded to 128 lanes).
- `bench/summarize.py`: KernelBench-style fast_p table (p 1.0, 1.25 vs `jit_topk`) and a flag for exact variants with speedup above 10x.
- Colab notebook: fused variant in the sweeps, block sweep for `pallas_score` and `pallas_fused_topk`, `approx_max_k` recall targets 0.80, 0.90, 0.95, 0.99.
- `CONTRIBUTING.md`, `docs/KERNEL_REVIEW.md` (Pallas TPU kernel review checklist), `NOTICE`.
- `.pre-commit-config.yaml` with ruff, file hygiene and a notebook-output check (`scripts/check_notebook_outputs.py`).
- Issue templates and a pull request template.
- Support matrix in `README.md`.

## 0.1.0

- V0 `numpy_ref`, V1 `jit_topk`, V2 `blockwise`, V3 `pallas_score`, A1 `approx_max_k`, stage-split timing.
- ann-benchmarks HDF5 loader, k-nn recall, roofline labels, plots.
- TPU v5e-1 results and `docs/RESULTS.md`.

[Unreleased]: https://github.com/maxpetrusenko/tpu-retrieval-bench/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/maxpetrusenko/tpu-retrieval-bench/releases/tag/v0.2.0
