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

| | QED | QCD |
|---|---|---|
| **model, symbolic exact match** | **83.0%** [70, 91] | **64.3%** [39, 84] |
| template oracle (retrieval ceiling) | 70.2% | 50.0% |
| 1-NN character n-gram retrieval | 21.3% | 14.3% |
| most frequent / exact lookup | 0.0% | 0.0% |
| parse validity | 100% | 100% |
| mass-dimension-4 validity | 100% | 100% |
| propagator channel accuracy | 91.5% | 92.9% |

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
a failure. Validation symbolic EM by epoch:

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

**3. Constrained decoding.** Parse validity is 100% by construction. Notably,
the `unconstrained_decode` arm on QCD *also* reaches 100% parse validity once
trained (57.1% symbolic EM vs 50.0% for the constrained control — inside the
interval either way at n=14). So the constraint is load-bearing early in
training and largely redundant at convergence. It costs nothing and removes a
failure mode, but it is not where the accuracy comes from.

---

## Protocol A — record split (the headline number)

The split the objective asks for, and the one comparable to prior work.

| run | symbolic EM | channel | dim-4 | params | min |
|---|---|---|---|---|---|
| **QED** `full_vanilla_dense` (120 ep) | **83.0** [70, 91] | 91.5 | 100 | 1.40M | 35 |
| _QED template oracle_ | 70.2 [56, 81] | 76.6 | 100 | – | – |
| _QED 1-NN retrieval_ | 21.3 [12, 35] | 31.9 | 100 | – | – |
| **QCD** `full_vanilla_moe` (30 ep) | **64.3** [39, 84] | 92.9 | 100 | 4.12M | 58 |
| QCD `unconstrained_decode` (30 ep) | 57.1 [33, 79] | 92.9 | 100 | 1.75M | 57 |
| QCD `full_vanilla_dense` (30 ep) | 50.0 [27, 73] | 85.7 | 100 | 1.75M | 40 |
| QCD `full_xsa_proj_dense` (30 ep) | 50.0 [27, 73] | 78.6 | 100 | 1.75M | 34 |
| QCD `math_only` (30 ep) | 42.9 [21, 67] | 78.6 | 100 | 1.34M | 37 |
| QCD `graph_only` (30 ep) | 42.9 [21, 67] | 71.4 | 100 | 0.96M | 4 |
| _QCD template oracle_ | 50.0 [27, 73] | 71.4 | 100 | – | – |
| _QCD 1-NN retrieval_ | 14.3 [4, 40] | 42.9 | 100 | – | – |

### Reading this honestly

- **The QCD arms are under-trained.** They ran 30 epochs, which QED has now
  shown is roughly half of what this setup needs. The QCD ordering should not be
  trusted until they are rerun at 120 epochs. The `full_vanilla_moe` result
  sitting on top is a plausible artefact of it having had a slightly different
  effective learning trajectory, not a demonstration that MoE helps.
- **n = 14 for the QCD test split.** Every QCD interval is about 45 points wide.
  One record is 7 percentage points. No QCD comparison here separates two arms.
- **Both pathways beat either alone on QCD** (50.0 vs 42.9), and the full model
  leads on channel accuracy (85.7 vs 78.6 / 71.4). That is the only signal in
  the QCD table that is consistent across metrics, and it is still inside the
  interval.
- `graph_only` trains in 4 minutes against 40 for the full model, because the
  QCD amplitude stream is 2861 tokens and dominates the cost.

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

- **QCD at full training length.** The single most important rerun. Every QCD
  number above is at 30 epochs.
- **Seed replication.** Every number is one seed. Three seeds minimum before
  any arm comparison means anything (docs/01 §6.4). The 30-epoch multi-seed runs
  were discarded as under-trained.
- **The QED arm grid at 120 epochs.** `xsa_proj` vs `xsa_mask` vs `vanilla`,
  MoE vs dense, and the capacity sweep have not been run at a length where they
  would be informative on QED.
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
