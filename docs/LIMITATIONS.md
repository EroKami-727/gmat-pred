# Limitations

What the current results do not support, stated plainly so a reviewer does not
have to find it. Generation G3 (per-planet models, 2026-08-02 onward).

## 1. Every headline number is single-seed

**Status: known gap, deliberately not closed yet. Disclose as a point estimate.**

The pruning economics table, the normalisation-collapse ablation, the tree-assist
result and the per-planet held-out metrics are all one run at seed 42. The split
is a deterministic function of `(n, seed)` (`src/ml/splits.py`), so re-running
reproduces them exactly — which is reproducibility, not stability. It says
nothing about how much the numbers move under a different partition.

What exists already: the *grouped* audits (LOTO, parameter-corridor) were run
across five seeds and reported std 0.000 everywhere, but those splits are
deterministic by target identity, so the seed only perturbs training noise. The
5-seed XGBoost confidence intervals in `reports/baselines/formal_ablation.json`
are genuinely seed-dependent and are the model for what the G3 table needs.

What closing it costs: per-planet training is ~50 s/planet on an RTX 4060, so 5
seeds x 7 targets is roughly an hour of training plus the economics re-run. It is
cheap. It is not done.

Until it is: report single-seed figures as point estimates. Do not attach a
confidence interval to them, and do not describe the ±0.000 from the grouped
audits as if it applied to the headline table.

WP4 in [`RESEARCH_PROPOSAL.md`](RESEARCH_PROPOSAL.md).

## 2. The outcome is determined at t=0

The simulator is deterministic. Outcome is a fixed function of the six injection
offsets, and the trajectory is their integral, so it cannot carry information the
parameters do not already have. Accuracy is flat from 10% to 40% observed.

This invalidates the "early warning from telemetry" framing on this dataset, and
it is why T0 screening beats T40. It is a property of the generator, not a
finding about trajectory prediction in general. WP1 exists to break it.

## 3. Per-timestep normalisation relies on a shared time base

Every mission of a given planet shares a time base, which is what makes
per-timestep standardisation work at all. Real campaigns do not have this. The
technique should be expected to degrade wherever mission durations vary
independently of the parameters being screened.

## 4. Cross-target generalisation is not solved

Leave-one-target-out collapses on Mars, Mercury and Moon (AUC at or below
chance), and Uranus/Venus rank well but are badly miscalibrated. The working
system is one model per planet — the opposite of transfer. Do not claim
zero-shot generalisation to an unseen target.

## 5. Simulator realism

Two/three-body dynamics, no execution error, no unmodelled accelerations, no
sensor noise, no navigation uncertainty.

User-built missions are *not* a separate limitation any more.
`src/api/mission_builder.py` constructs them with the same propagator and
feature code as the dataset, so they are in-distribution and scorable — verified
end to end, nominal missions scoring P(fail) 0.0001-0.0008 with no OOD flag. The
earlier heliocentric generator (`src/api/trajectory_gen.py`) produced
Sun-referenced elements against Earth-centric training data and drove every
generated mission to |z| ~ 1e13; it now supplies only planet constants and
Hohmann helpers to the creator UI, and generates nothing that gets scored.

The OOD detector still fires on genuinely extreme inputs — a 0.05 km/s TOI
offset is tens of sigma outside a corridor whose 1-sigma is 0.003 km/s — and
that is the intended behaviour, not a residual defect.

## 6. No real mission data

None. This is the largest external-validity gap and it needs a collaborator or a
public source of real trajectory telemetry (WP5).

## 7. Seven targets, not eight

Moon is excluded from all reported results by decision — it is a 6-day transfer
inside Earth's SOI sampled at 60 s, against seven heliocentric transfers of
127-13,419 propagation-days sampled at 15 h, and shares neither the cost
structure the economics are built on nor the dynamical regime. It remains
trained and served by the live simulator. See `EXCLUDED_TARGETS` in
`src/ml/planet_config.py`.

Reported results therefore cover 70,000 missions across seven targets, drawn
from an 80,000-mission eight-target generation.

## 8. Residual error concentrates in rare failure modes

The tree assist fixed Uranus `surface_impact` (sequence recall 0.000 → 1.000),
but the underlying defect is not understood: the sequence model could not reach a
signal demonstrably present in its own input, and mode-balanced resampling up to
45x did not move it. Whether that is optimisation, architecture or objective is
open (WP3). The fix is a fusion, not an explanation.

## 9. Superseded modules retained for provenance

`src/ml/regime_router.py` and `src/ml/per_target_calibration.py` implement the
G1/G2 regime-split approach that the per-planet rebuild replaced. They are kept
so the earlier results can be traced, and are no longer imported by the API. Do
not extend them.
