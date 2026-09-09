# 04 — Change review

**For review before commit.** Every change to the working tree, with its
evidence, its blast radius, and how to revert it. Nothing is committed.

**Scope.** The dual-pathway architecture, canonicalisation
(`data/canonical.py`), the Lark grammar, the Feynman graph builder, the model
and the decoder were read in full and **not modified**. Every change is in the
harness: seeding, metrics, the tensor pipeline, the sampler, the gates, docs.

**Verification.** `python tests/test_gates.py` → **47/47** (37 before this pass).

---

## Round 2 — what the first review changed

The first draft of this document proposed C1–C6. Review returned six findings.
Five were correct and are acted on below; the sixth changed how C5 is argued
rather than what it does.

| review finding | outcome |
|---|---|
| A run is not a function of `config + seed` | **Confirmed and fixed** — new C0, blocks everything else |
| `equal ⟹ structure` is not guaranteed; C1's identity can break | **Confirmed with a counterexample and fixed** — C1 revised |
| Over-budget held-out targets should score as misses, not raise | **Accepted** — C3 revised |
| `COMPLEXITY_CAP` guards the wrong quantity; fix it | **Accepted** — now C7, was "won't fix" |
| C5's accuracy justification doesn't isolate C5 | **Accepted and deleted** — C5 now rests on the a priori argument |
| The type-embedding comparison was never paired | **Confirmed** — §3 conclusion downgraded |

Two conclusions the first draft asserted are **withdrawn**; see §3.

---

## C0 — A run was not a function of its config and seed

**File:** `symba/data/pipeline.py`, `symba/experiment.py`,
`scripts/run_experiment.py`

**Defect.** `build` calls `set_seed(cfg.train.seed)` once. A job then trains
several arms against a *cached* bundle, and nothing reseeds before
`AmplitudeModel(...)`. So the initial weights depend on how many models,
dropout draws and shuffles were consumed earlier in the job — the same
`(arm, seed)` in a different position produces a different model. The loader's
shuffle generator has the same problem: by arm 2 it has already been advanced
by arm 1's epochs.

**Evidence.** Building `full_vanilla_dense` at seed 0, then rebuilding it after
one throwaway model, gives decoder-embedding weights differing by **0.53 max
absolute**. Not a rounding artefact.

This contradicts `config.py`'s own claim that "a run is fully described by this
config plus its seed", and it means arms differed in initialisation as well as
in the factor under test.

**Change.** `pipeline.seed_run(cfg, bundle)` resets the global RNG and the
bundle's shuffle generator; called immediately before model construction in
both run paths. `Bundle` now exposes `.generator`.

**Why it matters beyond reproducibility.** It gives common random numbers
across arms — arms at the same seed now start from the same weights and see the
same epoch order, so their difference is the factor under test. That is free
variance reduction on a corpus that has none to spare.

**Gate.** `test_G12_a_run_is_a_function_of_its_config_and_seed` builds one arm
twice around a throwaway and demands bit-identical `state_dict()`, then checks
the shuffle order resets too.

**Blast radius. Moves the numbers, and voids every existing one** — including
the measurements in §3 of this document's first draft.

**Revert.** Delete the two `seed_run` calls.

---

## C1 — `coefficient_exact` measured nothing (revised)

**File:** `symba/eval/metrics.py`, `scripts/report.py`

**Defect.** Defined as `equal and (pred_monomials == ref_monomials)`. The first
conjunct implies the second, so the metric was identically
`symbolic_exact_match` — verified across **46 of 46** archived runs.

**Change.** `structure_exact` (right channel **and** right monomial support)
over all predictions, and `coefficient_exact` conditional on structure, so
`symbolic EM = structure_exact × coefficient_exact`.

**What review caught.** That identity holds only if `equal ⟹ structure`, and it
did not. The reference is normalised `expand → together → cancel` by
`canonical.py`; a prediction went through `together` only. An equal-but-
uncancelled prediction therefore kept a denominator the reference does not
have. Constructed and confirmed:

```
ref   (s_12 + s_13) / s_14
pred  (s_12² − s_13²) / ((s_12 − s_13)·s_14)
      symbolically equal : True
      channel (strict)   : False     ← identity broken downward
```

The projective test proposed in review does not catch this one on its own —
`cancel(D_p/D_r) = (s_12 − s_13)`, not a number. The root cause is the missing
`cancel`. **Both fixes are applied:** `_as_fraction` now normalises predictions
the same way the target definition does, and `same_channel` is projective, so
denominators differing by a non-zero scalar count as one channel — which is
what a propagator channel means.

| case | before | after |
|---|---|---|
| uncancelled common factor | equal, **structure False** | equal, structure True |
| rational scalar `2N/2D` | equal, structure True | equal, structure True |
| genuinely wrong channel | not equal, structure False | not equal, structure False |

**Cost.** None measurable: QED baselines score in 3.2 s, QCD in 2.9 s, and every
baseline value is unchanged (QED oracle 70.2%, QCD 50.0%, channel 76.6 / 71.4).

**Gate.** `test_G14_equal_implies_structure` asserts
`count(equal ∧ ¬structure) == 0` over algebraically identical rewritings, and
checks the aggregate identity to 1e-9.

**Blast radius.** Does not move symbolic EM. Moves `channel_accuracy` upward
wherever a prediction was right up to a scalar.

---

## C2 — `raw_exact_match` was not raw

**File:** `symba/eval/metrics.py`, `symba/train/loop.py`, `scripts/report.py`

**Defect.** Named `raw_exact_match`, documented as "comparable to SYMBA",
actually comparing canonical *token sequences*. `RESULTS.md` leaned on a
comparability that does not hold.

**Change.** Renamed `sequence_exact_match`. `report.py` reads either key so
archived JSON still renders.

**Review's condition, accepted.** The rename removes a false comparability
without supplying a true one, so the `raw_target` arm is now scheduled — see
§6. It closes the largest open exposure in `00 §7.1`.

Because `to_prefix` is a normal form, this metric equals symbolic EM by
construction on canonical targets, and did in all 46 archived runs. It
separates only under `data.target="raw"` — which is exactly the arm now
scheduled.

**Blast radius.** Does not move symbolic EM. Breaking for external readers of
the old key.

---

## C3 — Length budgets were set by the test split (revised)

**File:** `symba/data/pipeline.py`

**Defect.** `lengths` was a maximum over train + val + test and served two
different purposes at once: sizing the positional tables, and — through
`lengths[2] + 4` — capping what the decoder may emit.

**Change.** The two are now separate quantities.

- **`decode_budget`** — what the decoder may emit. Training split only
  (`train_max + 8`). This is the one that had to be clean: it caps the thing
  under test.
- **`lengths`** — positional table sizes. The observed maximum plus a cushion.
  An embedding row no training gradient touches carries no information about a
  held-out target's *content*, and sizing tables from train alone would make a
  longer held-out sequence index off the end of the table.

**What review caught.** The first draft *raised* on an over-budget held-out
target. Wrong: under leave-one-template-out, a held-out class longer than
anything in training would kill the fold. A target the decoder structurally
cannot emit is a finding.

**Now:** it scores as a miss on its own — the model cannot produce it — and the
build counts it in `stats["unreachable_targets"]`. Currently `{val: 0, test: 0}`
on both theories.

**Gates.** `test_G10_decode_budget_comes_from_the_training_split_only` and
`test_G10_overlong_heldout_target_is_a_miss_not_a_crash`.

**Blast radius. Moves the numbers** (QED decode budget 109 → 113).

---

## C4 — The QCD encoder was fed `<unk>` in every record

**File:** `symba/data/pipeline.py`

**Defect.** `amp_to_segments` substitutes a `<diagrams>` placeholder for the
diagram sum. The amplitude vocabulary was built over `amp_tokens`, which never
contains that token, so **all 234 QCD records** fed the math encoder an
`<unk>`. Gate G7 could not see it because it measured `amp_tokens` rather than
`amp_segments` — the stream the model reads.

**Change.** Vocabulary and OOV report both built over `amp_segments`, which is
`[amp_tokens]` when segmentation is off.

**Evidence.** Before: 234 of 164,698 QCD encoder tokens unknown. After: 0.

**Blast radius. Moves the numbers, QCD only.** Every QCD figure in `RESULTS.md`
is void.

---

## C5 — Batch composition was frozen for the whole run (argument revised)

**File:** `symba/data/dataset.py`

**Defect.** `LengthBucketSampler` sorted globally once and reshuffled only the
*order in which batches were visited*. The partition of 277 training examples
into 18 batches was therefore fixed for all 120 epochs. That is deterministic
incremental gradient descent over a fixed partition — not SGD. It is
indefensible on its own terms and needs no accuracy evidence.

**Change.** Per-epoch reshuffle inside each diagram-count group, then sorted by
length in pools of eight batches. Grouping by diagram count first matters on
QCD: segments pad to an `(n_diagrams × longest_diagram)` rectangle, so mixing a
3-diagram record with a 15-diagram one costs about 4×. Measured after the
change, 3 of 36 QCD batches straddle a group boundary.

**What review caught.** The first draft justified this with a three-seed
accuracy comparison. That comparison did not isolate C5 — it moved together
with C3 and with uncontrolled initialisation (C0) — so **the justification is
withdrawn**, not weakened. The change stands on the argument above.

**Blast radius. Moves the numbers.**

**Revert.** Return `self.order` unchanged from `_epoch_order`.

---

## C6 — Gates

37 → 47. The ones that carry weight:

| gate | asserts |
|---|---|
| `G9_decoding_cannot_see_the_target` | blanking, randomising or deleting `batch["target"]` leaves beam output bit-identical; encoder memory unchanged |
| `G9_decoder_self_attention_is_causal` | randomising `target[5:]` moves logits after the cut, not before |
| `G12_a_run_is_a_function_of_its_config_and_seed` | two builds of one arm around a throwaway are bit-identical, shuffle order included |
| `G14_equal_implies_structure` | `count(equal ∧ ¬structure) == 0`, and the aggregate identity to 1e-9 |
| `G14_expansion_bound_catches_what_count_ops_misses` | the complexity guard bounds expansion, not written size |
| `G7_no_training_token_reaches_the_encoder_as_unk` | nothing the encoder reads encodes as `<unk>` |
| `G10_decode_budget_comes_from_the_training_split_only` | the emit cap is train-derived |
| `G10_overlong_heldout_target_is_a_miss_not_a_crash` | over-budget held-out targets are counted, not raised |
| `G14_per_record_vector_matches_the_aggregate` | the stored vector is what the headline sums |

---

## C7 — `COMPLEXITY_CAP` guarded the wrong quantity (was "won't fix")

**File:** `symba/eval/metrics.py`

**Defect.** The cap was applied to `count_ops(expr)` before `together`/`expand`.
`count_ops` measures the size of the *written* expression, which does not bound
the cost of expanding it. `OPERATOR_SLACK` admits roughly 60 binomials, so the
guard never bound at all:

| product of k binomials | `count_ops` | cap | expanded terms |
|---|---|---|---|
| 8 | 11 | 600 | 81 |
| 20 | 39 | 600 | 2048 |
| 60 | ~120 | 600 | 2⁶⁰ |

**The first draft left this documented and unfixed**, arguing it was a hole and
not a demonstrated cause. Review rejected that on two grounds, both correct:
the guard does not bind, and the reason given for not connecting it to the
stall — "not reproducible in isolation" — is exactly what C0 predicts. The
isolated rerun could not have been the same model at epoch 10, because model
initialisation was not a function of the seed. They are one bug.

**Change.** `expansion_terms(expr, cap)` — one pass over the expression tree,
product over `Mul`, sum over `Add`, short-circuited at the cap, with a negative
integer power counted as one factor because `expand` does not distribute over
`1/(a+b)`. Checked alongside `count_ops` in `_too_complex`.

**Evidence.** 20 distinct binomials: `count_ops` 39, bound 2048, rejected in
**0.0 ms**. A canonical target scores 1 and passes.

**Blast radius.** Can only turn a previously-hung or very slow scoring into a
counted bail-out. `complexity_bailouts` already reports the rate.

---

## C8 — Per-record outcomes are now persisted

**File:** `symba/eval/metrics.py`, `symba/train/loop.py`

Previously only `example_predictions[:3]` survived a run. Arms at a given seed
score the identical test records in the identical order, so `metrics.per_record`
now carries the `symbolic`, `structure` and `channel` boolean vectors, the
template key, and the dataset `index` for alignment.

That makes McNemar, error-overlap and the retrieval-margin diagnostic of
`03 §5.6` computable **on runs that already exist**, instead of needing a rerun
to ask a question nobody thought of at run time. It is a few dozen booleans per
run.

**Gate.** `test_G14_per_record_vector_matches_the_aggregate`.

---

## 3. Withdrawn conclusions

The first draft ran QED protocol A, 3 seeds × {control, `no_type_embedding`},
and drew two conclusions. **Both are withdrawn**, because C0 means those runs
did not control initialisation, and the archived numbers they were compared
against were produced by different code at a different RNG position.

| seed | control | `no_type_embedding` | n |
|---|---|---|---|
| 0 | 74.5% | 83.0% | 47 |
| 1 | 83.9% | 77.4% | 31 |
| 2 | 90.3% | stalled | 31 |

- ~~"C5 is cleared by three paired seeds."~~ Not paired. C5 stands on its own
  argument instead.
- ~~"The type embedding is not a win."~~ The *decision* — leave the default
  alone — is unchanged, because nothing supports changing it. But it is now an
  unsupported default rather than a measured null. Review also established that
  the archived 87.2% vs 83.0% was never paired either: `no_type_embedding` ran
  third in `QED_capacity.json`, after `capacity_64` and `capacity_256`, against
  a control that lives in a different file (`QED_record_long_seed0.json`).
  Confirmed by inspection of both files.

**What survives.** One within-run observation, which no between-run defect
touches: on every seed the variance sits in `structure_exact`
(80.9 → 87.1 → 96.8) while conditional `coefficient_exact` barely moves
(92.1 → 96.3 → 93.3). Finding the functional form is the unstable part; the
arithmetic is not. That is the structured head's target (`03 §2.3`).

`results/QED_record_typeembed.json` is kept as the artefact of a pre-C0 run and
should be regenerated, not cited.

---

## 4. Repository cleanup

Removed after verifying byte-identity against the copy kept in this repo:
`D:\SYMBA\docs\01–03`, `D:\SYMBA\ipynb files\` (4 notebooks), `D:\SYMBA\data\`,
four scratch directories, `proposal\~$OC_2026.docx` (Word lock file),
and `results/QED_loop2_typeembed.json` (aborted run, baselines only).

`docs/00_orientation.md` moved into the repo — the only unique file outside it,
and it documents this repo. `QED_loop3_typeembed.json` renamed
`QED_record_typeembed.json`.

`D:\SYMBA` now holds only `symba-amplitudes/`, `proposal/`, `Reference papers/`.
Two copies of the corpus was the hazard, not the disk space: nothing stopped
someone editing the one the pipeline does not read.

`checkpoint.py` `FORMAT_VERSION` 1 → 2. Version 1 checkpoints predate this pass
and are not loadable; they carry a different decode budget, so accepting them
silently would reproduce neither their numbers nor the current ones.

---

## 5. Still unexplained

**The stall.** Run 6 of six stopped progressing at its epoch-10 evaluation at
9.9 GB resident, no output for 80 minutes; killed. C7 is the most likely
mechanism and the isolated non-reproduction is explained by C0 — but that chain
is inference, not a demonstration, and no rerun has been done since. Prefer
`scripts/run_queue.py` for batches: resumable, and it releases metric caches per
arm in a `finally`, which `run_experiment.py` does not.

---

## 6. Order of work — do not reorder

Review's sequencing, adopted. Regenerating before the first two produces
numbers as unreproducible as the ones they replace.

1. **C0 first.** Done and gated.
2. **C8.** Done — the per-record vector must exist before runs are generated,
   or the paired tests need another rerun.
3. **Then regenerate.** `--batch core` and `--batch seeds`, both theories.
   Everything in `RESULTS.md` is void: QCD via C4, everything via C0.
4. **Then the `raw_target` arm** (C2's condition, `00 §7.1`) — one arm, and it
   closes the largest open exposure.

---

## 7. Deliberately out of scope

- **Tying `fc_out` to `token_embed + type_embed`** — the real inconsistency
  behind the type-embedding story. A modelling change; needs its own run, now
  possible to run properly because of C0.
- **The structured head** (`03 §2.3`) — the largest identified accuracy win,
  and §3 gives it a target: the structural half of the error.
- **Selection/reporting decoder mismatch** — checkpoints selected on greedy
  validation decode, reported on beam 4. Documented, unmeasured.
- **Protocol C** and **leave-one-template-out** — specified in `01 §5`, neither
  implemented. C3 no longer blocks the latter.
