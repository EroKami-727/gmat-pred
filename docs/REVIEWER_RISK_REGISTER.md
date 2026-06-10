# OrbitGuard Reviewer Risk Register

Use this before writing claims in the paper. Every high-confidence claim should
map to an experiment, and every known weakness should be disclosed or mitigated.

## Risk 1: Random Split Is Too Easy

Concern:

- XGBoost reaches near-perfect random-split performance.
- Initial-only XGBoost without context features still reaches 98.42% accuracy.

Evidence:

- `docs/PAPER_VALIDITY_AUDIT.md`

Mitigation:

- Report random split as an in-distribution result only.
- Include grouped and parameter-corridor holdout results.
- Do not claim broad generalization from random-split metrics.

## Risk 2: Transformer Is Not the Best Model

Concern:

- Transformer held-out test result is weaker than XGBoost.

Evidence:

- Transformer: 79.73% accuracy, F1 0.7447, ROC-AUC 0.9363.
- XGBoost full summary: 99.34% accuracy, F1 0.9899, ROC-AUC 0.9998.

Mitigation:

- Frame Transformer as a sequential neural baseline.
- Frame XGBoost trajectory-summary model as the strongest current classifier.
- Do not write model-centric claims that imply the Transformer is superior.

## Risk 3: Full Unseen-Planet Transfer Is Not Solved

Concern:

- Leave-one-target-out F1 collapses for Mars, Mercury, Moon, Neptune, and Venus.

Evidence:

- `docs/GROUPED_GENERALIZATION_AUDIT.md`

Mitigation:

- Claim only mixed unseen-target generalization.
- State that probability calibration fails on several held-out targets.
- Use parameter-corridor holdout as the more relevant operational test.

## Risk 4: Parameter-Corridor Holdout Has Failure Cases

Concern:

- Summary XGBoost performs well overall but weakens in sparse-success edge bins.
- `AOP` bin 1 is a major weak corridor.

Evidence:

- `docs/PARAMETER_HOLDOUT_AUDIT.md`

Mitigation:

- Report mean and per-bin metrics.
- Say trajectory summaries retain strong ranking performance, not universal
  perfect classification.
- Discuss operational thresholds as a calibration problem.

## Risk 5: Synthetic Fidelity

Concern:

- The pipeline uses simplified synthetic dynamics and calibrated targeting
  corridors, not high-fidelity GMAT/SPICE truth for every generated sample.

Mitigation:

- Describe the simulator as a controlled synthetic benchmark.
- Avoid claiming flight-readiness.
- Present OrbitGuard as an early-screening research framework.

## Risk 6: Cadence Confusion

Concern:

- Some folder names contain `15min`, but interplanetary telemetry is 54,000
  seconds = 15 hours.

Mitigation:

- State cadence explicitly in Methods.
- Never describe interplanetary data as 15-minute cadence.
- Keep `docs/RESEARCH_LEDGER.md` updated.

## Current Strongest Contribution Framing

1. A calibrated multi-target synthetic benchmark for early trajectory outcome
   prediction from Moon through Neptune.
2. A leakage-audited feature pipeline using physics/context features rather
   than explicit outcome columns.
3. Evidence that random-split prediction is easy and that initial-condition
   corridors are highly separable.
4. Evidence that trajectory summaries generalize better than initial-only
   features under parameter-corridor holdout.
5. An honest comparison showing XGBoost as the strongest current model and the
   Transformer as a neural sequential baseline.

## Claims To Avoid

- "Transformer outperforms all baselines."
- "OrbitGuard generalizes robustly to unseen planets."
- "The model is ready for operational mission cancellation."
- "Interplanetary telemetry is sampled every 15 minutes."
