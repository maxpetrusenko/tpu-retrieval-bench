# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub security advisories:
https://github.com/maxpetrusenko/tpu-retrieval-bench/security/advisories/new

Please do not open a public issue for a suspected vulnerability. Include the affected
version or commit, reproduction steps and the impact you see.

## Disclosure timeline

This is a single-maintainer project; the timeline below is what to expect.

| step | target |
|---|---|
| Acknowledge the report | within 7 days |
| Triage and confirm or decline, with reasons | within 30 days |
| Fix released, or a published advisory with a mitigation | within 90 days of the report |
| Public disclosure | with the fix or advisory, coordinated with the reporter; at 90 days at the latest unless a fix needs longer and the reporter agrees |

## CVE handling

Confirmed vulnerabilities are published as GitHub security advisories on this repository.
When the impact justifies it, the advisory requests a CVE identifier through GitHub, which is
a CVE Numbering Authority. The advisory names the affected and fixed versions and credits the
reporter unless they ask not to be named.

## Supported versions

| version | supported |
|---|---|
| `main` (latest commit) | yes |
| latest release tag (0.2.x once tagged) | yes |
| older releases and untagged commits | no |

Fixes land on `main` and ship in the next release tag.

## Scope

This repository is a benchmark harness. Reports about the CI workflows, dependency
pinning and the benchmark code are in scope.

## How changes are reviewed

This is a solo-maintainer project, so there is no second human reviewer. Changes go through a
pull request into `main`, and branch protection requires the CI checks `lint-test (py3.12)`,
`lint-test (py3.13)` and `audit` to pass before a merge. The `sast (bandit)`, `fuzz
(hypothesis)`, `package` and CodeQL checks also run on every pull request. An AI coding agent
writes and reviews part of the diffs, and its review is not a substitute for an independent
human one. Scorecard's Code-Review check stays low for that reason; adding a second maintainer
is the fix, and it has not happened yet.

## Automated checks

Dependencies are locked (`uv.lock`) and audited in CI (`pip-audit`). CodeQL and bandit
(medium severity and up) scan the Python code, Hypothesis fuzz tests and an atheris harness
exercise the result validator, and OpenSSF Scorecard tracks the supply-chain posture. Actions
are pinned to commit SHAs and updated by dependabot. Releases carry a build provenance
attestation, `SHA256SUMS` and a CycloneDX SBOM (README, "Verify a release").

Known exception: the Colab notebook and the manual `tpu-smoke` workflow install with
`pip -c notebooks/constraints.txt`, which pins versions but not hashes, because `jax[tpu]`
comes from a Google-hosted wheel index. Scorecard's Pinned-Dependencies check may flag it.

## Self-hosted runner

One CI job (`test`, label `[self-hosted, vps-builder]`) runs on a self-hosted runner. Its
isolation:

- The runner is an ephemeral VPS builder with no secrets: no deploy keys, cloud credentials or
  tokens are stored on it, and workflows pass no repository secrets to it. Its checkout uses
  `persist-credentials: false` and `contents: read`.
- The job triggers on `push` to `main` and on manual `workflow_dispatch` only, never on
  `pull_request`. Fork pull requests therefore never reach the runner, and only users with
  write access can dispatch the workflow.
- The job runs lint, tests and the result validator, which need no network credentials.
- If the runner is ever suspected of compromise, remove it in the repository settings and
  rebuild the VPS; nothing on it needs rotating.
