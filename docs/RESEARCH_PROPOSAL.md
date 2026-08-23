# Research Proposal — OrbitGuard

**Draft, 2026-08-02. Numbers refreshed 2026-08-16** after removing two
selection biases from the economics evaluation (see §7). Measured on the local
80,000-mission generation, of which the seven-target study set is 70,000, and
reproducible from this repository. Claims are separated into what the current
data supports and what it does not; the full list is in
[`LIMITATIONS.md`](LIMITATIONS.md).

---

## 1. Summary

Monte Carlo trajectory campaigns spend most of their compute on missions that
were doomed at injection. OrbitGuard studies how much of that compute can be
recovered by a learned screen, and where in the pipeline the screen should sit.

The headline finding is a negative one that reframes the problem. On a
deterministic simulator, a six-feature classifier over launch parameters prunes
**64.5% of propagation cost while discarding 0.76% of good missions** — and a
sequence model reading 40% of the trajectory does strictly worse on the same
budget (38.3% saved). The trajectory is the integral of the injection state and
carries no additional information.

The second contribution is methodological, and it is the part most likely to
transfer: a **failure mode in which grouped feature scaling silently destroys a
deep model, while scale-invariant baselines report that everything is fine.**

---

## 2. Contributions

**C1 — Grouped-normalisation collapse, and why baselines hide it.**
Sharing one feature scaler across heterogeneous groups compresses within-group
variation below what gradient descent can amplify. A Transformer trained this
way degenerated to a per-group constant (Venus P(fail) = 0.020910 for every
mission, AUC ≈ 0.5) while XGBoost on *identical data with identical scaling*
scored AUC 1.0000. Validation AUC looked healthy at 0.955–0.997 because it
measured between-group ranking, not within-group discrimination.

We give the diagnostic (compare a tree against the network under identical
preprocessing; check per-group prediction variance), the mechanism, and a
controlled three-condition ablation (`src/ml/norm_ablation.py`,
`reports/normalisation_ablation.json`).

**Corrected 2026-08-16.** An earlier version of this section cited "Mars val AUC
0.939 → 0.998 from normalisation alone", from a within-planet comparison that
was never reproducible. The rebuilt ablation refutes it: pooling a single
target's own timesteps costs at most 0.005 AUC. The collapse requires the scaler
pooled *across targets*, and it is selective — Venus collapses to a constant
(AUC 0.9999 → 0.6037, output std 3.11e-05) while Mercury, Mars and Jupiter lose
0.0003–0.0925 AUC and keep working. Severity does not track the compression
ratio, so no predictive rule is claimed. The baseline-blindness half holds on
every target: XGBoost on identical input scores 0.9996–1.0000 under all three
conditions (spread ≤ 0.0002). See `RESEARCH_LEDGER.md`, "C1 Rebuilt, and Partly
Refuted".

**C2 — A quantified pruning result, with the honest conclusion.**
Compute charged in propagation-days across a ~100× cost range (Mercury 127 d to
Neptune 13,419 d), at a fixed 99% failure-recall operating point:

| Screen | Compute saved | Good missions destroyed | Failure recall |
|--------|---------------|-------------------------|----------------|
| T0 — launch parameters, before propagating | **64.5%** | **0.76%** | 99.06% |
| T40 — telemetry Transformer at 40% | 38.3% | 0.21% | 98.87% |
| Cascade — T0 where confident, else T40 | 65.0% | 1.62% | 99.94% |

T0 also predicts *how* a mission fails at 0.96–0.99, matching the sequence
model. Logistic regression on the same features scores AUC 0.49 — chance — so
nonlinearity is essential even though sequence modelling is not.

Thresholds are fitted on validation and reported on an untouched test split; the
recall column is what the operating point achieves out of sample rather than a
target met by construction. An earlier version of this table was produced with
test-set thresholding and an evaluation set that overlapped the sequence model's
validation split — correcting both moved the headline by less than 0.1 pp
(`src/ml/prune_economics.py` documents the protocol).

**C3 — Rare-mode failure of a sequence model against a demonstrable signal.**
Uranus `surface_impact` (119 of 6,611 failures) had sequence recall 0.000, while
a tree on the *identical normalised input window* separated it at AUC 1.000.
Resampling the mode up to 45× did not move recall at all. This is an
optimisation limit, not an information limit, and it is measurable rather than
speculative. Fusing a per-planet tree assist at the decision window raised
overall held-out F1 from 0.9960 to **0.9981** (recall 0.9991, 5 false negatives
in 8,400 missions).

**C4 — Dataset and reproducible harness.** 70,000 GMAT-derived missions across
seven interplanetary targets — drawn from an 80,000-mission eight-target
generation, with Moon excluded from the study as a non-interplanetary regime — per-planet models, a mission builder that reuses the dataset's
own propagator so user-defined missions are in-distribution, and an end-to-end
simulator.

---

## 3. What the current data cannot support

Stated plainly so reviewers do not have to find it:

- **Early-warning framing.** Accuracy is flat from 10% to 40% observed. The
  outcome is fixed at t=0, so "watching the trajectory" is not what produces the
  result.
- **Cross-target generalisation.** Leave-one-target-out collapses on
  Mars/Mercury/Moon, and the working system is one model per planet — the
  opposite of transfer.
- **Realism.** A deterministic two/three-body simulator with no execution noise,
  no unmodelled forces, and no sensor error. Every mission of a planet shares a
  time base, which is what makes per-timestep standardisation work at all; that
  is a property of the generator, not of real campaigns.
- **Real mission data.** None.

---

## 4. Proposed work

### WP1 — Make the sequential question well-posed (highest value)

The current dataset cannot distinguish "the model reads the trajectory" from
"the model reads the initial condition", because they are the same thing. Break
that by regenerating a subset with **mid-flight stochasticity**: execution error
on a deterministic-sequence midcourse correction, unmodelled acceleration, and
observation noise on the telemetry the model sees.

The prediction is falsifiable: T0 accuracy must degrade with perturbation
magnitude while T40 holds. The crossover point where sequence models start
paying for themselves is itself the result, and it is the experiment that
converts C2 from a negative result into a positive one.

*Cost:* regeneration of ~10k missions per condition; the Numba propagator makes
this hours, not days.

### WP2 — Generalise C1 beyond astrodynamics

Reproduce grouped-normalisation collapse on two public grouped time-series
datasets with heterogeneous group scales, and characterise when it occurs as a
function of between-group / within-group variance ratio. A predictive rule —
"collapse when the ratio exceeds X" — would make C1 a usable check rather than
an anecdote.

### WP3 — Why the sequence model cannot reach a present signal

C3 is currently an observation. Establish whether it is optimisation
(loss-landscape, gradient starvation from a 1.8%-frequency mode), architecture
(CLS pooling discarding a localised cue), or objective (binary head dominated by
the majority mode). Probe the trunk representation for linear separability of
the rare mode; if separable there, the defect is in the head.

### WP4 — Statistical rigour on the headline table

Multi-seed confidence intervals on the economics table, per-planet cost
sensitivity, and a risk/savings frontier rather than a single operating point.
Currently single-seed.

### WP5 — Extend to real trajectory data

The strongest external-validity step, and the one that requires a collaborator
or a public source of real mission telemetry.

---

## 5. Venue framing

Two viable papers, and they should not be the same paper:

1. **Methods / negative results venue** — C1 + C3. "Scale-invariant baselines
   can certify a broken deep model." Self-contained today; stronger with WP2.
2. **Aerospace systems / simulation venue** — C2 + C4, with WP1 as the core
   experiment. Honest form: *where* in a simulation pipeline a learned screen
   belongs, with cost accounting, concluding that input-space screening
   dominates for deterministic simulators and identifying the regime where it
   stops dominating.

Submitting the sequential framing without WP1 invites the obvious rejection: the
task is solved by six tabular features and the paper's own baseline shows it.

---

## 6. Reproduction

```bash
export ORBITGUARD_DATA=/path/to/merged_all_v2   # see docs/ENVIRONMENT.md

# Per-planet extracts (one streaming pass) + models + tree assist
setsid nohup ./run_pipeline.sh > /dev/null 2>&1 &
python -m src.data_collection.recover_mission_ids
python -m src.ml.train_assist --all
python -m src.ml.recalibrate

# Headline tables
python test_ml.py --limit 1200          # accuracy + failure mode + created missions
python -m src.ml.prune_economics        # compute saved vs missions destroyed
```

Full decision history, including the two analyses invalidated by a positional
join and their corrections, is in [`RESEARCH_LEDGER.md`](RESEARCH_LEDGER.md).

---

## 7. Evaluation protocol, and a correction to it

The economics table was originally produced under two selection biases. Both
have been removed and the table above is the corrected run; the headline moved
by less than 0.1 pp.

**Thresholds were fitted on the test labels.** The 99%-failure-recall operating
point was chosen using the test split's own labels, which no deployed screen can
do. Thresholds are now fitted on validation and applied unchanged to test, so
the reported recall is a measured out-of-sample quantity (99.06% weighted)
rather than 99% by construction.

**The evaluation set contained the sequence model's validation split.** The
economics script drew its own 70/30 partition at seed 42 while the trainer used
70/15/15 at the same seed — the same permutation, so the "test" set was exactly
the model's validation plus test split. Half the missions the T40 screen was
scored on were the ones its checkpoint and abort threshold had been selected on.
`src/ml/splits.py` is now the single definition of the partition, shared by the
trainer, the economics script and the test harness.

Both biases flattered **T40** — the screen this analysis concludes against — so
the negative result survives its own correction. That is the strongest form the
claim can take, and it is why the correction is reported rather than quietly
applied: reviewers should be able to see that the conclusion does not depend on
the evaluation being generous to the alternative.

Single-seed remains an open limitation (WP4); see
[`LIMITATIONS.md`](LIMITATIONS.md) §1.
