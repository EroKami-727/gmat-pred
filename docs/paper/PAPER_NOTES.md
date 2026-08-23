# Notes on the draft

Editorial commentary on `OrbitGuard_paper_draft.docx`. The paper states claims;
this file states how much each one is worth, what is missing, and where a
reviewer will push. Read this before sending the draft to anyone.

The `.docx` is **generated**, not hand-written — `docs/paper/build_paper.py`
reads the JSON artifacts in `reports/` and formats them. Do not edit the `.docx`
directly for anything quantitative: the numbers will silently diverge from the
experiments, which is the exact failure this project already had once (three
model generations of contradictory figures across `docs/`, see
`docs/README.md`). Edit the prose in `build_paper.py`, or re-run the experiment.

## Claim → evidence map

| Claim in the paper | Backed by | Confidence |
|---|---|---|
| Grouped normalisation collapses the network to a constant | `reports/normalisation_ablation.json` | **Measured.** Controlled: same architecture, seed, split; only pooling differs. |
| Timestep pooling is survivable, group pooling is not | same | **Measured**, and this decomposition is new — the original incident report blamed scaling in general. |
| A tree is invariant to the normalisation that kills the network | `reports/baseline_invariance.json` | **Measured.** Tree AUC moves ≤0.0002 across all three conditions. |
| Prediction variance is a working diagnostic | same two artifacts | **Measured** as a correlate. Not validated as a detector on any other dataset. |
| The sequence model cannot reach a rare mode a tree separates | `reports/rare_mode_sweep.json` | **Measured** on one target, one rare mode. |
| Input-space screening dominates telemetry screening | `reports/prune_economics.json` | **Measured**, and independently re-verified after removing two selection biases. |
| Logistic regression on the six features is at chance | `auc_logreg_t0` in the economics artifact | **Measured.** |
| Architecture description (d_model 128, 8 heads, 4 layers, Pre-LN, CLS) | `src/ml/model.py` | **Verified against source.** |
| 70,000 missions / 7 targets / 10,000 each | `data/per_planet/*.npz`, `planet_config.py` | **Verified.** |

## What changed from the earlier framing, and why

Two things changed once the experiment was actually built, and both weaken the
original claim. They are stated here rather than buried because the draft is
worth less if you learn them from a reviewer.

**1. The within-planet comparison does not reproduce the collapse.**
`RESEARCH_PROPOSAL.md` described C1 as "sharing one feature scaler across
heterogeneous groups", supported by a table comparing a global RobustScaler
against per-timestep z-scoring **within a single planet** (Mars val AUC
0.939 → 0.998). That table was never reproducible — no script produced it — and
the rebuilt experiment contradicts it: pooling one target's own timesteps costs
at most a few thousandths of AUC. All four targets survive it.

The collapse requires pooling across *targets*. The paper now makes that
sharper claim, with a three-condition ablation that separates the two kinds of
pooling. This is a better result than the original, but it is a different one,
and the ledger's version is wrong.

**2. The collapse is selective — one target in four.**

| Target  | grouped AUC | per-timestep AUC | P(fail) std, grouped |
|---------|------------|------------------|----------------------|
| Venus   | 0.6037     | 0.9999           | 3.11e-05 — collapsed |
| Mercury | 0.9069     | 0.9994           | 4.25e-01             |
| Mars    | 0.9422     | 0.9994           | 4.06e-01             |
| Jupiter | 0.9997     | 1.0000           | 4.75e-01             |

Only Venus degenerates to a constant. The others lose 0.0003–0.09 AUC and keep
discriminating. Severity does not track the signal ratio cleanly either:
Jupiter's grouped signal (0.0247) is *lower* than Mercury's (0.0484) yet Jupiter
loses almost nothing, because its task is separable enough to survive heavy
attenuation. So compression is necessary but not sufficient, and the "collapse
when the ratio exceeds X" rule the proposal hoped for is **not supported**. The
draft says so explicitly.

**Scope caveat now in the paper.** The ablation isolates normalisation alone —
one model per target, only the statistics pooled. The production incident also
shared a single model across the group, which compounds the damage. The measured
numbers are a lower bound on the original failure, not a reproduction of it.

### Is it still a paper?

Yes, but a more careful one than the proposal implied. The defensible core is:
a routine preprocessing choice silently destroyed one target completely, and the
standard baseline check reported AUC 1.0000 on that same target under that same
preprocessing. One catastrophic instance plus demonstrated baseline blindness is
a legitimate cautionary methods contribution. It is *not* a general law about
grouped normalisation, and the draft does not claim one.

The honest weakness: n=4 targets, one seed, one collapse event. Reviewer 2 will
say "you found a bug on Venus". The answers are (a) the ablation is controlled
and the mechanism is measured, not inferred, and (b) the baseline-blindness
result holds on all four targets, not just the one that collapsed — that half is
not anecdotal. Strengthening (a) needs more seeds and a second dataset.

## What is NOT evidenced

- **Related work — Section 7 is a stub.** It cannot be generated from the
  repository. This is the single largest gap and it is yours to fill: you need
  positioning against per-group/per-instance normalisation, shortcut learning
  and Clever-Hans effects, the tabular-vs-deep literature on why trees stay
  competitive, and learned early termination for simulation. Without it the
  paper is not submittable regardless of how good the experiments are.
- **Single seed.** Everything is seed 42. Reproducible ≠ stable. Cheap to fix
  (~1 hour): re-run `norm_ablation` and `rare_mode_sweep` across 5 seeds and
  report intervals. Do this before submission; a reviewer will ask whether a
  collapse at AUC 0.60 is a seed artifact, and right now the honest answer is
  "we don't know".
- **One dataset.** The mechanism is described in terms of a variance ratio that
  is not astrodynamics-specific, but it has been shown on one corpus. The paper
  says this. A methods venue will still want a second demonstration — ideally a
  public grouped time-series dataset with heterogeneous group scales.
- **The rare-mode cause.** Section 4 establishes that the failure is an
  optimisation limit rather than an information limit, and stops there. It does
  not say *why*. The cheapest next probe is linear separability of the rare mode
  in the trunk representation: separable there ⇒ the defect is in the head.

## A number that changed: 45x vs 19.2x resampling

The ledger and `train_assist.py`'s docstring say the rare mode was oversampled
"up to 45x" without recovering it. The measured sweep reports **19.23x** at
`mode_alpha=1.0`. Both are right under different denominators: 45x is the ratio
to the *majority* failure mode (4113/91 = 45.2), 19.2x is the factor relative to
*uniform* sampling over the training set, which is what the sampler actually
applies. The paper reports the latter and labels it, because it is the quantity
that describes what the optimiser saw. If you quote 45x anywhere, say against
what.

The conclusion is unchanged and if anything stronger than the prose claimed:
recall on the rare mode is exactly 0.0000 at every alpha, while a tree on the
identical window reaches AUC 1.0000 and recall 1.0000.

## Where a reviewer will push

1. **"Isn't this just a bug you had?"** The defence is the controlled ablation
   plus the decomposition — it is a reproducible property of the preprocessing,
   not an incident. Lead with Table 1, not with the story.
2. **"Why would anyone pool a scaler across groups?"** Because it is the default
   when you fit one `RobustScaler` on a concatenated training set, which is what
   almost every tutorial pipeline does. Make that explicit early; the finding is
   only interesting if the mistake is natural.
3. **"Your fix is just per-group normalisation, which is known."** True and the
   paper should not claim the fix is novel. The contribution is the *detection*
   problem: that the standard sanity check certifies the broken configuration.
   Keep the emphasis there.
4. **"Single seed."** No defence. Fix it.
5. **"Section 5 says the whole application doesn't need the model."** Deliberate.
   It bounds the practical claim and pre-empts the obvious objection. A reviewer
   who finds it themselves is much worse than one who is told.

## Venue

Written for a **methods / negative-results** audience — the collapse and the
detection failure are the product, and the astrodynamics is the setting. Do not
send this to an aerospace venue: Section 5 concludes their application does not
need the method, and Sections 3–4 are about optimisation pathology.

The aerospace paper is a *different* paper (the economics plus the dataset), and
it needs WP1 — regenerating data with mid-flight stochasticity — before it has a
positive result to report rather than a negative one.

## Open decisions for you

- **Affiliation** is a placeholder in `build_paper.py`.
- **Authorship** — `rohitmichael-alt` has 15 commits in the repository's history
  and the earlier dataset generation. Decide the author list before circulating.
- **Dataset release.** C4 in the proposal offers the corpus as a contribution.
  The mission tables are ~71 GB; a release needs a hosting plan and probably a
  downsampled public subset. The per-planet `.npz` extracts (~70 MB each) are a
  reasonable candidate.
- Whether to fold Section 5 into the paper at all, or cut it and keep the paper
  purely methodological. Current draft keeps it, on the grounds that it is
  honest and pre-empts an obvious objection — but it does dilute the focus.
