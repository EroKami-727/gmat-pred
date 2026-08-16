# OrbitGuard Reviewer Risk Register

**Current as of generation G3 (per-planet models, 2026-08-02).** Rewritten from
the G1 version, which was steering claims using the original multi-planet
Transformer's numbers (79.73% accuracy) — a model that no longer exists. The G1
risks that were really about *split design* rather than about that model are
carried forward unchanged, because they still bite.

Use this before writing claims. Every high-confidence claim should map to an
experiment, and every known weakness should be disclosed or mitigated.

## Risk 1: Random split is too easy

**Still current.**

- XGBoost reaches near-perfect random-split performance.
- Initial-only XGBoost without context features still reaches 98.42% accuracy.
- G3 makes this sharper, not milder: six launch parameters reach AUC
  0.9961-1.0000 per planet *before any propagation happens*.

Evidence: [`PAPER_VALIDITY_AUDIT.md`](PAPER_VALIDITY_AUDIT.md),
`reports/prune_economics.json`.

Mitigation:

- Report random split as an in-distribution result only.
- Include grouped and parameter-corridor holdout results.
- Do not claim broad generalisation from random-split metrics.

## Risk 2: The sequence model does not earn its place

**Rewritten.** The G1 form of this risk — "the Transformer is weaker than
XGBoost, frame it as a baseline" — is no longer factually right: the G3
per-planet Transformer reaches held-out F1 0.9981. The underlying risk is real
but the reason changed, and the new reason is worse for a sequential framing.

Concern: a six-feature model on launch parameters, evaluated at zero propagation
cost, matches the sequence model on both outcome (AUC 0.9961-1.0000) and failure
mode (0.96-0.99). Charged in propagation-days, screening at T0 saves 64.5% of
compute against T40's 38.3%. Accuracy is also flat from 10% to 40% observed.

The cause is structural: the simulator is deterministic, so the outcome is a
fixed function of the six injection offsets and the trajectory is their
integral. It cannot carry information the parameters do not already have.

Mitigation:

- Do not write "early trajectory prediction saves compute" as the headline. The
  paper's own baseline refutes it on this dataset.
- State the T0-vs-T40 comparison explicitly and early. A reviewer will find it
  otherwise, and it is much better as a contribution than as a discovered flaw.
- Logistic regression on the same six features scores AUC 0.49 — chance. The
  task needs nonlinearity, just not sequence modelling. Say both halves.

## Risk 3: Full unseen-planet transfer is not solved

**Still current.**

- Leave-one-target-out F1 collapses for Mars, Mercury, Moon, Neptune and Venus.
- The working system is one model per planet — the opposite of transfer.

Evidence: [`GROUPED_GENERALIZATION_AUDIT.md`](GROUPED_GENERALIZATION_AUDIT.md),
[`ERROR_ANALYSIS.md`](ERROR_ANALYSIS.md).

Mitigation:

- Claim only mixed unseen-target generalisation.
- Distinguish the two failure modes: Uranus/Venus rank well but are miscalibrated
  (fixable by calibration), while Mars/Mercury/Moon have AUC at or below chance
  (genuine regime shift, 1-43 sd feature displacement).
- Use parameter-corridor holdout as the more relevant operational test.

## Risk 4: Parameter-corridor holdout has failure cases

**Still current.**

- Summary XGBoost performs well overall but weakens in sparse-success edge bins.
- `AOP` bin 1 has the worst calibration in the entire audit (ECE 0.410).

Evidence: [`PARAMETER_HOLDOUT_AUDIT.md`](PARAMETER_HOLDOUT_AUDIT.md).

Mitigation: report mean *and* per-bin metrics; PR-AUC is the honest metric in
low-success bins, not raw accuracy.

## Risk 5: Synthetic fidelity

**Still current, and the most likely rejection reason for an aerospace venue.**

Deterministic two/three-body dynamics, no execution noise, no unmodelled forces,
no sensor error, and every mission of a planet shares a time base. That last
property is what makes per-timestep standardisation work at all, and it is a
property of the generator rather than of real campaigns.

Mitigation: describe the simulator as a controlled synthetic benchmark, avoid
claiming flight-readiness, and disclose that determinism is what drives the
T0-dominance result. WP1 in [`RESEARCH_PROPOSAL.md`](RESEARCH_PROPOSAL.md)
exists to break exactly this assumption.

## Risk 6: Cadence confusion

**Still current.** Some folder names contain `15min`, but interplanetary
telemetry is 54,000 s = 15 hours. Moon is 60 s. State cadence explicitly in
Methods; never describe interplanetary data as 15-minute cadence.

## Risk 7: Single-seed headline results

**New in G3.** The economics table, the normalisation-collapse ablation and the
tree-assist result are all seed 42, single run. See
[`LIMITATIONS.md`](LIMITATIONS.md); this is a known gap, not an oversight, and
must be disclosed as a point estimate rather than dressed up with a spurious
interval.

## Risk 8: Scope of the target set

**New in G3.** Eight targets were generated; seven are reported. Moon is
excluded by decision (`EXCLUDED_TARGETS` in `src/ml/planet_config.py`) because
it is not an interplanetary transfer and shares neither the cost structure nor
the dynamical regime. Disclose the exclusion and the reason in Methods — an
unexplained missing target reads as a dropped inconvenient result.

## Current strongest contribution framing

1. **Grouped-normalisation collapse, and why scale-invariant baselines hide it.**
   A shared scaler across heterogeneous groups compresses within-group variation
   below what gradient descent can amplify; the network degenerates to a
   per-group constant while XGBoost on identical data reports AUC 1.0000.
2. **A quantified pruning result with an honest conclusion** — input-space
   screening dominates telemetry screening for a deterministic simulator, with
   cost accounting in propagation-days across a ~100x cost range.
3. **A rare-mode optimisation failure**, measurable rather than speculative:
   Uranus `surface_impact` at sequence recall 0.000 against a tree at AUC 1.000
   on the identical normalised input window.
4. A calibrated multi-target synthetic benchmark and a reproducible harness.

## Claims to avoid

- "The Transformer outperforms all baselines." (G1 reason is stale; the T0
  comparison is the real objection.)
- "Early trajectory prediction saves compute." Not on this dataset.
- "OrbitGuard generalises robustly to unseen planets."
- "The model is ready for operational mission cancellation."
- "Interplanetary telemetry is sampled every 15 minutes."
