# Documentation index

Read this before quoting a number out of `docs/`.

This directory accumulated across **three model generations**, and results from
all three are still present. They disagree with each other — not because any of
them is wrong, but because they describe different models. Nothing said so, so
the same quantity appears at three values in three files and it is impossible to
tell which is current without reading the git log.

Every superseded document now carries a banner naming its generation. This page
is the authority on which is which.

## Generations

### G1 — XGBoost audits and the original multi-planet Transformer
**2026-06-10 → 06-16. Superseded.**

The first paper-grade audit pass: random-split baselines, leave-one-target-out,
parameter-corridor holdout. The sequence model of this era is the original
80K-dataset Transformer at **79.73% accuracy, F1 0.745, AUC 0.936**.

- [`PAPER_VALIDITY_AUDIT.md`](PAPER_VALIDITY_AUDIT.md)
- [`GROUPED_GENERALIZATION_AUDIT.md`](GROUPED_GENERALIZATION_AUDIT.md)
- [`PARAMETER_HOLDOUT_AUDIT.md`](PARAMETER_HOLDOUT_AUDIT.md)
- [`STATISTICAL_AUDIT_SUMMARY.md`](STATISTICAL_AUDIT_SUMMARY.md)
- [`REVIEWER_RISK_REGISTER.md`](REVIEWER_RISK_REGISTER.md) — *partly current, see below*

### G2 — Local 80K reproduction, multi-planet Transformer
**2026-06-21 → 06-23. Superseded.**

An independent regeneration of the dataset on this machine, and a longer
training run of the multi-planet sequence model: **87.67% accuracy, F1 0.838,
AUC 0.984** at 50 epochs. Also the domain-generalisation (target upweighting)
experiment and the per-target error analysis.

- [`PAPER_READY_SUMMARY.md`](PAPER_READY_SUMMARY.md)
- [`STATISTICAL_AUDIT_SUMMARY_LOCAL.md`](STATISTICAL_AUDIT_SUMMARY_LOCAL.md)
- [`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md)
- [`AI_CONTEXT.md`](AI_CONTEXT.md)

### G3 — Per-planet dual-head models, tree assist, T0/T40 economics
**2026-08-02 → current. This is what the paper describes.**

One model per target instead of shared regime models, per-timestep
normalisation, a failure-mode head, a fused per-planet tree assist, and the
pruning-economics analysis. Held-out **F1 0.9981**, and the finding that
launch-parameter screening at T0 beats telemetry screening at T40.

- [`RESEARCH_LEDGER.md`](RESEARCH_LEDGER.md) — full decision history, including
  the analyses that were invalidated and redone. Start here.
- [`RESEARCH_PROPOSAL.md`](RESEARCH_PROPOSAL.md) — contributions and proposed work.
- [`LIMITATIONS.md`](LIMITATIONS.md) — what the current results do not support.
- [`ENVIRONMENT.md`](ENVIRONMENT.md) — interpreter, dependencies, data paths.
- [`../README.md`](../README.md) — progress report and quick start.

Generation-independent (infrastructure, not results):

- [`NUMBA_JIT_PROPAGATOR.md`](NUMBA_JIT_PROPAGATOR.md)

## Contradictions you will otherwise trip over

**Sequence-model accuracy appears as 79.73%, 87.67% and 99.81%.** Three
different models: G1's multi-planet Transformer, G2's longer run of it, and G3's
per-planet models. All three numbers are correct for their generation. Only the
G3 figure describes anything that currently exists on disk.

**"XGBoost is the strongest model; the Transformer is a baseline."** True in G1
and G2, where the multi-planet Transformer genuinely underperformed trees. In G3
the per-planet Transformer reaches F1 0.9981, so the statement as written is no
longer true — but the *conclusion it was guarding* survives in a stronger form:
a six-feature XGBoost on launch parameters, evaluated before any propagation
happens, matches the sequence model on both outcome and failure mode at zero
propagation cost. Do not claim sequence modelling earns its place on this
dataset. Claim it for the T0-vs-T40 reason, not the G1 reason.

**Seven targets or eight?** Eight were generated (80,000 missions, 10,000 each).
The study set is **seven** — Moon is excluded by decision, documented at
`EXCLUDED_TARGETS` in `src/ml/planet_config.py`. Reported results therefore
cover **70,000 missions across seven interplanetary targets**. Moon remains
trained and served by the live simulator; it is out of the paper, not out of the
product.

**Cadence.** Some older folder names contain `15min`. Interplanetary telemetry
is 54,000 s = **15 hours**. Moon is 60 s. This has always been a naming
artifact; never describe the interplanetary data as 15-minute cadence.

**Dataset paths.** G1/G2 docs reference `data/merged_through_neptune_15min/` and
`/media/Data/.../merged_all_v2/`. The current dataset root is resolved from
`$ORBITGUARD_DATA` — see [`ENVIRONMENT.md`](ENVIRONMENT.md).
