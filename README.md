# Physics-Informed Models for Squared Amplitude Calculation

Structure-aware sequence models that predict the squared amplitude
$\overline{|\mathcal{M}|^2}$ from a Feynman diagram and its amplitude, for the
SYMBA / ML4SCI project.

The map being learned is exactly deterministic — there is one right answer and
it is computable. The difficulty is not in the physics but in the *encoding*:
MARTY emits nested, unsimplified, dummy-index-laden strings, and a
sequence model asked to reproduce them byte for byte is being handed a lossy,
redundant, non-canonical version of a function it could otherwise be given
cleanly. This repository is built around that observation.

---

## What is here

| stage | module | what it does |
|---|---|---|
| S1 | `symba/data/load.py` | parse and validate; 4 fields, 4 external legs, one record per non-blank line |
| S2 | `symba/data/normalize.py` | dummy-index normalisation, keyword stripping |
| S3 | `symba/data/canonical.py` | sympy canonicalisation of the target + mass-dimension check |
| S4 | `symba/data/ast_parse.py`, `graph.py` | Lark LALR grammar → AST; Feynman diagram → edge list |
| S5 | `symba/data/serialize.py` | typed prefix serialisation, and its inverse |
| S6 | `symba/data/vocab.py` | vocabulary from the **training split only** |
| — | `symba/data/dataset.py` | dynamic padding, padding masks, no truncation |
| S7 | `symba/model/` | dual-pathway encoders + cross-attention decoder |
| S8 | `symba/eval/decode.py` | beam search with length normalisation and grammar constraints |
| S9 | `symba/eval/metrics.py`, `baselines.py` | the seven metrics, and the baselines they must beat |

Design rationale, the bug list this replaces, and the argument for each change
live in [`docs/`](docs/).

## Results

QED **83.0%** symbolic exact match [70, 91] against a 70.2% retrieval ceiling
and a 21.3% 1-NN baseline, with 100% parse and mass-dimension validity. Under a
template split, where whole functional forms are held out, the model and every
baseline score **0%**. See [RESULTS.md](RESULTS.md).

## Quick start

```bash
pip install -r requirements.txt
```

**1. Verify the pipeline.** The gate tests are the reason to trust anything
below; nothing else should be run until they pass.

```bash
python tests/test_gates.py
```

**2. Train.** Jobs run one at a time and any `(arm, seed)` already in the
output file is skipped, so an interrupted batch resumes rather than restarts.

```bash
python scripts/run_queue.py --batch core
```

See what a batch will do, or what is already done:

```bash
python scripts/run_queue.py --list
```

**3. Read the results.**

```bash
python scripts/report.py results/*.json
```

**4. Predict with a trained checkpoint.**

```bash
python scripts/predict.py --list
```

```bash
python scripts/predict.py --checkpoint checkpoints/QCD_record_long__full_vanilla_dense__seed0.pt --input data/Symba/QCD/QCD-2-to-2-diag-TreeLevel-0.txt --limit 5
```

Every finished run writes `checkpoints/<job>__<arm>__seed<n>.pt`, carrying the
weights, the config, the three vocabularies and the sequence lengths — enough
to rebuild the model and reproduce its decoding with no access to the training
data. `scripts/predict.py` applies the identical S2–S5 preprocessing, and when
the input line carries a ground-truth `sq_amp` it scores the prediction with
the same symbolic-equivalence test used in training.

---

## The five changes that matter

### 1. The model is actually fed the parsed representations

The previous pipeline built the AST, built the graph, printed shape assertions
for both — and then constructed its tensors from the **raw infix strings**:

```python
graph_str  = item.get('vertices', '')   # raw
math_str   = item.get('amp',      '')   # raw
target_str = item.get('sq_amp',   '')   # raw
```

with loaders built from the un-cleaned record lists. Nothing the grammar or the
graph builder produced ever reached the network. Every claim resting on
algebraic hierarchy or interaction topology was **untested — not disproved,
untested**.

Gate `G9` now asserts the wiring directly: perturb the raw `amp` field and the
model input must not change; perturb `amp_tokens` and it must.

### 2. Nothing is truncated

The old budgets were 120 / 168 / **65** bytes. Measured against them, **100% of
amplitudes and 100% of targets were truncated in both theories** — the reported
"exact match" was exact match on the first ~63 characters.

Canonicalising the target removes the length problem rather than clipping it:

| | raw median / p90 / max | canonical median / p90 / max |
|---|---|---|
| QED `sq_amp` | 127 / 205 / 210 | 201 / 212 / 221 |
| QCD `sq_amp` | 421 / 2409 / **2872** | 205 / 247 / **258** |

QCD's tail was MARTY emitting unsimplified output, not intrinsic complexity.
Serialised as typed prefix, the target has a **31-symbol vocabulary (QED) / 30
(QCD)** and a maximum of **103 / 134 tokens** — measured on this corpus by
`tests/test_gates.py`. Sequences over budget raise; they are never clipped.

### 3. The graph is a graph

`OffShell A(V_1)` inside vertex `V_1` was matched against its own vertex,
producing the self-loop `('V_1','V_1','A')`. The internal line joining the two
vertices — the thing that determines the denominator channel — was never
represented, so the "topology" pathway encoded two disconnected vertex bags.

Propagators are now resolved by matching the off-shell particle type *across*
vertices, giving a real edge with distinct endpoints. All 594 records build a
connected graph (`G6`).

### 4. Honest evaluation

- **Three disjoint splits.** The old code used the test set as the validation
  set for early stopping, checkpointing, and reporting.
- **Free-running metrics only.** The old per-epoch "exact match" was
  teacher-forced — the ground-truth prefix was fed in at every step. That number
  is logged here but is never called exact match.
- **Reported at the selected checkpoint**, not as a maximum over epochs of a
  test statistic on n = 36.
- **Selection on symbolic exact match**, not on cross-entropy. With
  $\varepsilon = 0.1$ smoothing the loss floor is 0.8778 and the old run reached
  0.8828: the selection signal was $5\times10^{-3}$ wide while exact match was
  still climbing.
- **Wilson intervals everywhere**, because n is 36 and 24.
- **Baselines reported alongside every number**: most-frequent, exact lookup,
  1-NN character n-gram retrieval, and a template oracle.

### 5. Constrained decoding

The target vocabulary is ~30 symbols with known operator arities, so the set of
legal next tokens is computable from the decoder's own prefix state. Any token
that cannot complete a well-formed expression within the remaining length budget
is masked. **Parse validity is 100% by construction**, and whole classes of
error become unreachable.

---

## Why the numbers will look different

This corpus is much smaller than it appears. Mapping each mass symbol to the leg
index of its first occurrence and canonicalising collapses it to:

| | records | distinct raw targets | **distinct template classes** |
|---|---|---|---|
| QED | 360 | 162 | **30** |
| QCD | 234 | 118 | **11** |

(Both counts are reproduced by this pipeline; see `symba/data/splits.py`.)

Under a random record-level split, essentially every test record belongs to a
template seen in training, so that number measures *substitution within a seen
template*. A 1-NN retriever with no training at all is a strong baseline on it.
Two split protocols are therefore reported together:

- **Protocol A (record)** — the headline number, comparable to prior work.
- **Protocol B (template)** — whole functional forms held out. The honest
  generalisation number.

They will differ substantially. That gap is a result, not an embarrassment.

Note also that removing truncation and switching to the canonical target
**makes the task harder**: the model now has to emit a complete ~100-token
expression instead of the first 63 bytes of one. A lower number than the old
notebooks reported is expected and is not a regression — the old number was
measuring something else.

---

## Repository layout

```
symba/
  config.py            one dataclass tree; a run is config + seed
  data/                S1-S6, plus splits and the tensor pipeline
  model/               embeddings, attention arms, FFN arms, the model
  train/               loop, schedule, selection
  eval/                decode, metrics, baselines
scripts/               run_experiment.py, report.py, overnight_*.sh
tests/test_gates.py    the fourteen gates of docs/01 SS7
notebooks/             the original exploratory notebooks, kept for provenance
data/Symba/            QED and QCD tree-level corpora
docs/                  architecture design, bug list, proposed changes
```

Notebooks import; they do not define. Everything that was a notebook cell is a
module with a test.

## Ablation arms

Selected with `--arms`; each differs from the control in one factor.

| arm | what it changes |
|---|---|
| `full_vanilla_dense` | control: both pathways, standard attention, dense FFN |
| `graph_only` / `math_only` | modality ablation — the pathway is *not built*, not zeroed |
| `full_xsa_proj_dense` | XSA as implemented previously: project the output off its own value vector |
| `full_xsa_mask_dense` | XSA as *argued* previously: mask the attention diagonal before softmax |
| `full_vanilla_moe` | MoE FFN with load-balancing loss and utilisation logging |
| `capacity_64` / `capacity_256` | capacity sweep |
| `no_type_embedding` | removes the grammar-derived type channel |
| `role_filler` / `tpr_binding` | additive vs multiplicative role binding |
| `unconstrained_decode` | grammar constraints off — isolates their contribution |
| `raw_target` | raw MARTY string instead of the canonical form |

The two XSA arms are separate on purpose: projecting the output off its own
value vector and masking the attention diagonal are different operators, and the
motivation given for XSA describes the second while the previous code
implemented the first.

## Known limitations

- **QCD amplitudes are long.** The input AST reaches 2861 tokens (a sum over
  diagrams), so QCD runs are far slower than QED. Per-diagram encoding is the
  principled fix and is not implemented yet.
- **MoE is expected to do nothing here.** 324 examples against 30 target
  functions — capacity is not the binding constraint. It is instrumented so the
  claim can be *measured* rather than asserted.
- **The structured head is not implemented.** Predicting the denominator
  channel, monomial support, and coefficients directly is likely the largest
  remaining accuracy win, especially on QCD.
- Trained on CPU at small capacity; the numbers are not a scaling result.

## License

MIT — see [LICENSE](LICENSE).
