# Results

All numbers are **free-running** decoding (no teacher forcing), at the
checkpoint selected on a **validation split disjoint from test**, with 95%
Wilson intervals. Nothing here is a maximum over epochs.

Run on CPU, small capacity (d_model 128, 2+2+2 layers). Regenerate the tables
with:

```bash
python scripts/report.py results/*.json --markdown --out RESULTS.md
```

---

## Headline

Both theories at 120 epochs, full canonical target, free-running beam decode.
**Mean over three seeds** — a seed changes both the weight initialisation and
the split, so a single run is not the result.

| | QED | QCD |
|---|---|---|
| **model, symbolic exact match** | **84.6%** (sd 4.1) | **96.5%** (sd 2.9) |
| per-seed values | 83.0 / 80.6 / 90.3 | 100.0 / 92.9 / 96.8 |
| template oracle (retrieval ceiling) | 70.2% | 50.0-78.6% |
| 1-NN character n-gram retrieval | 21.3% | 14.3-25.0% |
| most frequent / exact lookup | 0.0% | 0.0% |
| parse validity | 100% | 100% |
| mass-dimension-4 validity | 100% | 100% |
| propagator channel accuracy | 91.5% | 100% |

The model **beats the template oracle on both theories**. That matters more
than the raw percentage: the oracle is what pure retrieval achieves when it is
told the right template class, so exceeding it is the evidence that something
other than interpolation is happening. It also gets the propagator channel right
more often than the oracle does (91.5% vs 76.6% on QED).

**This is not comparable to the old notebook's 94.4%.** That number was exact
match on the first 63 bytes of a truncated target, selected as a maximum over
epochs, on a test set that was also the validation set. These numbers are for
emitting a complete ~93-token canonical expression, scored once at a selected
checkpoint on a held-out split.

---

## What actually mattered

Ranked by measured effect:

**1. Training length, by a mile.** The first grid ran 30 epochs and looked like
a failure. QCD went from 50.0% at 30 epochs to **100.0% at 120**; QED from
0% to 83.0%. Validation symbolic EM by epoch (QED):

| epoch | 20 | 30 | 40 | 60 | 70 | 90 |
|---|---|---|---|---|---|---|
| QED val symbolic EM | 8.3% | 36.1% | 63.9% | 66.7% | 75.0% | 75.0% |

At 30 epochs the model sits *below* the 21.3% retrieval baseline; at 70 it is
above the oracle. Everything from the 30-epoch grid was discarded as measuring
the compute budget rather than the architecture. Selection early-stopped at
epoch 100, best at 70.

**2. Canonicalisation and the removal of truncation.** These changed the task
before any model ran. The target went from a 2872-character worst case to 258,
from a 260-symbol byte alphabet to a 31-symbol typed vocabulary, and from 100%
of targets truncated to none.

**3. Per-diagram encoding.** Encoding each Feynman diagram separately cut QCD
encoder attention work 10.7x, dropped the longest attended sequence from 2859
tokens to 239, and took a training epoch from 107 s to 45 s. That is what made
120-epoch QCD runs affordable on CPU at all, and 120 epochs is what took QCD
from 50% to 100%.

**4. Constrained decoding — but not for the reason expected.** Parse validity
is 100% by construction, yet `unconstrained_decode` matches the control exactly
on QED (83.0% both) and also reaches 100% parse validity once trained. The
constraint is load-bearing early in training and redundant at convergence. It
costs nothing and removes a failure mode; it is not where the accuracy comes
from.

---

## Protocol A — record split (the headline number)

The split the objective asks for, and the one comparable to prior work.

| run | symbolic EM | channel | dim-4 | params | min |
|---|---|---|---|---|---|
| **QED** `full_vanilla_dense` (120 ep) | **83.0** [70, 91] | 91.5 | 100 | 1.40M | 35 |
| _QED template oracle_ | 70.2 [56, 81] | 76.6 | 100 | – | – |
| _QED 1-NN retrieval_ | 21.3 [12, 35] | 31.9 | 100 | – | – |
| QED `unconstrained_decode` | 83.0 [70, 91] | 87.2 | 100 | 1.40M | 28 |
| QED `full_xsa_proj_dense` | 80.9 [67, 90] | 91.5 | 100 | 1.40M | – |
| QED `full_vanilla_moe` | 80.9 [67, 90] | 89.4 | 100 | 3.77M | 49 |
| QED `full_xsa_mask_dense` | 78.7 [65, 88] | 87.2 | 100 | 1.40M | 31 |
| **QCD** `full_vanilla_dense` | **100.0** [78, 100] | 100 | 100 | 1.75M | 338 |
| QCD `math_only` | 85.7 [60, 96] | 100 | 100 | 1.34M | 41 |
| QCD `graph_only` | 85.7 [60, 96] | 92.9 | 100 | 0.96M | 12 |
| _QCD template oracle_ | 50.0 [27, 73] | 71.4 | 100 | – | – |
| _QCD 1-NN retrieval_ | 14.3 [4, 40] | 42.9 | 100 | – | – |

### Reading this honestly

- **The QCD 100% was one seed of three.** On seed 0 it is 14/14; the other two
  seeds give 92.9% and 96.8%, so the honest figure is 96.5% (sd 2.9). Do not
  quote the 100%.
- **Claim 1 holds on both theories, and only multiple seeds showed it.**
  Mean symbolic EM over three seeds:

  | | graph only | AST only | graph + AST |
  |---|---|---|---|
  | QED | 51.6 (sd 12.9) | 72.6 (sd 1.6) | **84.6** (sd 4.1) |
  | QCD | 86.3 (sd 2.2) | 91.8 (sd 4.6) | **96.5** (sd 2.9) |

  Monotone on both, and on QED the gaps are several times the seed spread.
  On seed 0 alone QCD looked like a tie between the single-modality arms
  (85.7 / 85.7); across three seeds the ordering is consistent. This is the
  strongest result in the project and it is the one the proposal is about.
- **No QED arm separates from another.** 78.7 to 83.0 across `vanilla`,
  `xsa_proj`, `xsa_mask` and `moe`, with intervals ~22 points wide, all
  overlapping. That is exactly what docs/03 SS7 predicted for both XSA
  (item 11, "approximately 0 at this n") and MoE (item 12, "approximately 0,
  possibly negative"). Reporting it as a null result is the finding.
- **Constrained decoding does not drive the accuracy.** `unconstrained_decode`
  matches the control at 83.0% and still reaches 100% parse validity once
  trained. The constraint guarantees well-formedness early in training and
  costs nothing, but it is not where the number comes from.
- `graph_only` trains in 12 minutes against 338 for the full model, because the
  QCD amplitude pathway dominates the cost even after segmentation.

---

## Protocol B — template split (the honest generalisation number)

Whole functional forms are held out: no test record's template class appears in
training.

| | QED | QCD |
|---|---|---|
| split (train/val/test) | 198/60/102 | 174/12/48 |
| classes (train/val/test) | 17/4/9 | 6/2/3 |
| model symbolic EM | **0.0%** | **0.0%** |
| every baseline, including the oracle | 0.0% | 0.0% |
| parse validity | 98.8% | 100% |
| **mass-dimension-4 validity** | **0.0%** | 100% |

**Nothing generalises to an unseen functional form.** Not the model, not
retrieval, and not the oracle — the oracle is 0% by construction, since there is
no same-template training record to retrieve.

The interesting part is the QED diagnostic: 98.8% of outputs are well-formed
expressions and 0% are dimensionally consistent. Asked for a formula it has
never seen, the model produces something syntactically valid and physically
impossible. That is exactly the separation between "wrong arithmetic" and "wrong
physics" that the mass-dimension invariant was added to expose.

This is the number that says what the corpus can and cannot support. Protocol A
measures substitution within a seen template — a real skill, and the stated
objective — but it is not evidence of learned physics. Protocol B says there is
currently none.

*Caveat on the QCD protocol-B split:* 11 template classes cannot be divided
into three meaningful groups. 6/2/3 classes is the best available and the test
side is 3 classes. Leave-one-template-out (docs/03 §5.6) is the right design and
is not implemented.

---

## What is not yet measured

Honest gaps, in rough order of how much they would change the picture:

- **Seed replication.** Every number above is one seed. Three seeds minimum
  before any arm comparison means anything (docs/01 SS6.4), and with intervals
  this wide the QED arm ordering is currently noise. Running
  (`--batch seeds`).
- **The capacity sweep.** `capacity_64` / `capacity_256`, plus
  `no_type_embedding`, `role_filler` and `tpr_binding`, have not been run at
  120 epochs (`--batch capacity`).
- **A QCD test set worth the name.** n=14 makes 100% and 93% indistinguishable.
- **Leave-one-template-out**, which would give protocol B an n equal to the
  class count instead of a single arbitrary grouping.
- **The structured head** (docs/03 §2.3) — predicting denominator channel,
  monomial support, and coefficients directly. Given that channel accuracy is
  already 91.5% while full symbolic EM is 83.0%, the remaining errors are in the
  numerator, which is what that head targets.
- **NTK / CKA measurements** (docs/03 §4.4), which would let architecture
  claims be made on 277 training points instead of a 47-record test set.
- **Per-diagram encoding for QCD** (docs/01 §4.2), which is what makes the
  2861-token amplitude tractable.

## Reproducing

```bash
python tests/test_gates.py                      # 29/29 must pass first
python scripts/run_experiment.py --theory QED --protocol record --seeds 0 \
  --arms full_vanilla_dense --epochs 120 --patience 25 --eval-every 10
python scripts/report.py results/*.json
```
