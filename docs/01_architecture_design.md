# 01 — Architecture Design

**Blueprint for the system to be built. No code is written until §7 (the gate
table) is agreed and nobody can find another hole in it.**

Objective, stated exactly: maximise squared-amplitude prediction accuracy on a
held-out portion of the QED and QCD corpora, for a single unified system that
subsumes what are currently four disconnected notebooks (Task 1.2 tokenisation and
Task 2 model are **one pipeline**, not two deliverables).

---

## §0 The evidence base

Every design decision below is forced by a measurement on
`D:\SYMBA\data\Symba`. These are facts, not opinions.

### 0.1 Size and integrity

QED 360 records, QCD 234 records; 100% parse into 4 ` : `-separated fields;
4 external legs in every record. Distinct `amp` strings: 360 / 234 (all unique).
Distinct `sq_amp` strings: 162 / 118.

### 0.2 Lengths (characters)

| field | median | p90 | max |
|---|---|---|---|
| QED `vertices` | 107 | 121 | 129 |
| QED `amp` | 246 | 487 | 517 |
| QED `sq_amp` | 127 | 205 | 210 |
| QCD `vertices` | 107 | 116 | 125 |
| QCD `amp` | 650 | 4053 | **4734** |
| QCD `sq_amp` | 421 | 2409 | **2872** |

Against the current caps (120 / 168 / **65** bytes): **100% of `amp` and 100% of
`sq_amp` are truncated in both theories.**

### 0.3 Algebraic form of the target

Every target in both corpora is

$$\big|\mathcal{M}\big|^2=\frac{N\!\left(\{s_{ij}\},\{m_X\},c\right)}{D},\qquad c\in\{e,g\}$$

| | QED | QCD |
|---|---|---|
| free symbols in the whole corpus | 17 | 14 |
| monomials in the expanded numerator | median 5, **max 5** | median 5, **max 8** |
| expanded numerator length (chars) | median 103, max 112 | median 105, max 158 |
| factored denominator length (chars) | median 35, max 37 | median 33, max 35 |

**The numerator is mass-dimension homogeneous of degree exactly 4 in 360/360 and
234/234 records — 100%, no exceptions** (weights $m_X\!\to\!1$, $s_{ij}\!\to\!2$,
`reg_prop`$\to\!2$, coupling $\to\!0$). Exact, free, checkable. Used nowhere today.

### 0.4 Canonicalisation removes the length explosion

`expand → together → cancel`:

| | raw med / p90 / max | canonical med / p90 / max |
|---|---|---|
| QED `sq_amp` | 127 / 205 / 210 | 201 / 212 / 221 |
| QCD `sq_amp` | 421 / 2409 / **2872** | 205 / 247 / **258** |

QCD's tail is MARTY emitting unsimplified nested output, not intrinsic
complexity: p90 falls **10×**, max falls **11×**. Canonicalisation also merges
**118 → 99** distinct QCD targets: 19 raw strings are algebraically identical to
another raw string, so raw-string exact match currently *penalises correct answers*.

### 0.5 Canonical prefix serialisation — the target budget

| | vocab | median tokens | p90 | **max** |
|---|---|---|---|---|
| QED target | **30** | 108 | 120 | **125** |
| QCD target | **28** | 110 | 135 | **143** |

Complete QED target vocabulary:
`* + 0…9 NEG ^ e m_b m_c m_d m_e m_mu m_s m_t m_tt m_u reg_prop s_12 s_13 s_14 s_23 s_24 s_34`

**A target budget of 160 tokens covers 100% of both corpora.** Compare: a
260-symbol byte vocabulary truncating 100% of targets at 63.

Input side, `amp` through the existing Lark grammar → AST prefix:

| | vocab | median | p90 | max |
|---|---|---|---|---|
| QED `amp` | 61 | 139 | 258 | 262 |
| QCD `amp` | 172 | 359 | 2257 | **2859** |

The QCD *amplitude* tail is real, not an artefact — it is a sum over diagrams.
This is the one place long context genuinely matters (§4.2).

### 0.6 The target space is 30 and 11 functions

Map each mass symbol to the leg index of its first occurrence
(`m_e → M1`, `m_mu → M3`, …), then canonicalise:

| | records | distinct raw | distinct canonical | **distinct after mass→leg-role** |
|---|---|---|---|---|
| QED | 360 | 162 | 162 | **30** |
| QCD | 234 | 118 | 99 | **11** |

Class sizes — QED: `36,36,36, 18×9, 12×3, 6×3, 3×12`; QCD: `60,36,36,36,30, 6×6`.

The rational prefactor tracks charge and colour ($\tfrac14 e^4$ for $e\mu$,
$\tfrac{4}{81}e^4$ for $u\bar t$), so flavour substitution is exact only *within* a
charge/colour class — and the corpus already contains those variants, which is
precisely why only 30 classes exist.

### 0.7 Baselines nobody has reported

200 random 90/10 record splits, current truncated regime (target = `sq_amp[:63]`):

| | QED | QCD |
|---|---|---|
| test prefix appears **verbatim** in train | 78.8% ± 6.9 | 76.0% ± 9.0 |
| exact lookup on truncated graph input | 22.5% ± 6.3 | 30.6% ± 9.3 |
| **1-NN char 3–5-gram TF-IDF, zero training** | **40.1% ± 7.7** | **12.6% ± 6.0** |
| most-frequent prefix | 1.1% | 10.9% |
| reported v3 beam exact match | 94.4% (34/36) | 66.7% (16/24) |

### 0.8 Consequence for the objective

Under a record-level split, ≈100% of test records fall in a template class seen in
training. "Accuracy on the validation portion" is therefore, as currently defined,
mostly a measurement of **substitution within a seen template**. The blueprint
below maximises that number honestly *and* instruments the system so the number
can be decomposed into what is retrieval and what is composition (§5, §6). Both
are needed: the first is the stated goal, the second is what makes the result
mean anything.

---

## §1 Design principles

**P1 — One pipeline, one config, one entry point.** No parallel data paths. The
single most damaging property of the current code is that four representations
were built and the model consumed a fifth. There must be exactly one route from
raw line to tensor, and a test that asserts it.

**P2 — Nothing silently truncates, drops, or falls back.** Every lossy operation
raises or is counted and logged. `except Exception: item['amp_ast'] = ['<UNK>']`
is banned; a parse failure is a build failure.

**P3 — Representation before capacity.** Fixing the representation (canonical
prefix, 30-symbol vocab, 160-token budget) changes the problem from
"reproduce 63 truncated bytes" to "emit a bounded, well-typed algebraic object."
No architectural change matters until this is done.

**P4 — Every claim is a run.** Ablation hooks are not enough; each claim needs a
seed-replicated run with a pre-declared metric, an n, and a confidence interval.

**P5 — Physics as constraint, not decoration.** Invariants that hold in 100% of
the data (mass dimension 4, rational-function form, dummy-index relabelling) are
enforced or measured, not merely cited in prose.

---

## §2 Pipeline overview

```
raw line
  │  S1  parse & validate            → Record
  ▼
Record (interaction, vertices, amp, sq_amp)
  │  S2  normalise                   → dummy indices, whitespace, keywords
  ▼
  │  S3  canonicalise target         → sympy N/D, algebraic canonical form
  ▼
  │  S4  structural parse            → amp AST · Feynman graph
  ▼
  │  S5  serialise                   → prefix token lists (typed)
  ▼
  │  S6  vocabulary (train split ONLY) → int IDs
  ▼
Tensors  (graph_ids, amp_ids, target_ids, masks, meta)
  │  S7  model                       → logits
  ▼
  │  S8  decode (constrained)        → token list
  ▼
  │  S9  deserialise & compare       → metrics
```

Each arrow is a stage with an explicit contract (§7).

---

## §3 Data stages in detail

### S1 — Parse and validate
Split on ` : `, require exactly 4 fields. **Assert count == file line count.**
Currently true (360/360, 234/234) but must be asserted, not assumed. Record the
source file and tree level as metadata — needed for split design.

### S2 — Normalise
- Dummy-index normalisation (`%\sigma_18923 → %\sigma_d1`), first-occurrence
  order, per expression. This already exists and is correct; it simply is not
  applied on the path the model sees.
- Vertex/interaction keyword stripping.
- **Invariant to assert:** every `\` in the corpus is preceded by `%`
  (verified: 5712/5712). Re-check at load; fail loudly if a new file breaks it.

### S3 — Canonicalise the target
Parse `sq_amp` with sympy (`i → I`, `^ → **`), then:

```
e            = expand(parse(sq_amp))
N, D         = fraction(cancel(together(e)))
N            = expand(N)            # ≤5 (QED) / ≤8 (QCD) monomials
D            = factor(D)            # ≤37 chars
canonical    = (N, D)
```

Store **both** the raw string and the canonical form. Raw is needed to report a
SYMBA-comparable metric; canonical is the training target.

*Design note.* This changes what "the answer" is, from MARTY's particular output
string to the mathematical object it denotes. That is a real change and it must be
declared, not hidden — §6.1 specifies reporting both metrics side by side so the
change is auditable.

### S4 — Structural parse
- **AST:** existing Lark LALR(1) grammar. It parses 594/594 pairs with zero
  failures. Keep it. Failures become build failures (P2), with a logged pattern.
- **Graph:** the existing `FeynmanGraph` is *almost* right but has a real bug —
  `OffShell A(V_1)` inside `V_1` produces the self-loop `('V_1','V_1','A')`
  instead of an edge to the partner vertex. See `02_problems_and_flaws.md` §2.4.
  The graph builder must emit a genuine edge list with propagator endpoints
  resolved by matching particle type across vertices.

### S5 — Serialise (typed)
Prefix (Polish) notation, digits split, with a **token type tag** carried
alongside each token id:

```
type ∈ {OP, MASS, MANDELSTAM, COUPLING, DIGIT, REGPROP, STRUCT, SPECIAL}
```

The type channel costs 8 embedding rows and buys: constrained decoding (§S8),
a free interpretability axis (which types route to which expert, §6.4), and a
diagnostic partition of the error (§6.2). This is cheap and should not be skipped.

### S6 — Vocabulary
Built from the **training split only**. Special ids fixed low
(`PAD=0, SOS=1, EOS=2, UNK=3`), digits `0–9` fixed next, then sorted symbols for
determinism. Vocabulary is serialised with the checkpoint.

**Assert:** vocabulary construction happens strictly after the split. The current
tokeniser notebook builds it over the whole corpus first — a real leak.

### Length budgets (from §0.5, with headroom)

| tensor | budget | coverage |
|---|---|---|
| `target_ids` | **192** | 100% of QED and QCD canonical targets |
| `graph_ids` | **192** | 100% of vertices+interaction |
| `amp_ids` (QED) | **320** | 100% |
| `amp_ids` (QCD) | **3072** | 100% — or per-diagram split, §4.2 |

Padding is **dynamic per batch**, with a length-bucketed sampler; the fixed-length
tensor is abandoned. Truncation, if it ever occurs, raises.

---

## §4 Model

The dual-pathway skeleton is kept. The changes are corrections and additions, not
a redesign — the architecture was never actually tested, so it is not yet refuted.

### 4.1 Encoders
- Two independent stacks (graph, math) — unchanged.
- **Padding masks everywhere.** Currently absent; this is not optional.
- Positional information: per-pathway `RoleFillerProjection` (v3 already fixed the
  shared-instance bug present in the baseline).
- Pre-LayerNorm residual blocks — unchanged.
- Dropout inside attention and FFN, not only after GBST.

### 4.2 The QCD amplitude length problem
QCD `amp` reaches 2859 AST tokens. Three options, to be decided by experiment:

1. **Full sequence, 3072 budget.** Quadratic attention at 3072 × 8 heads is about
   9.4M attention entries per layer per sample — feasible at batch 4–8 on one GPU.
   Simplest; establishes the reference.
2. **Per-diagram encoding.** The amplitude is a sum over diagrams; split on
   top-level `+`, encode each diagram independently with shared weights, then
   attend over the set of diagram summaries. Reduces the quadratic term to
   $\sum_d L_d^2 \ll (\sum_d L_d)^2$ and injects the correct permutation
   invariance over diagrams. Physically motivated and cheaper.
3. **Canonicalise the amplitude too.** The amp is dominated by index bookkeeping;
   the same `cancel/together` compression that gave 11× on the target may apply.
   Must be measured before it is assumed.

Option 2 is the principled one and is the recommendation, with option 1 as the
control that shows what option 2 costs or buys.

### 4.3 Tokenisation layer
GBST as implemented is a learned multi-scale local mixer, not Charformer's block
selection (see `02` §3.1). Two decisions:

- With the canonical prefix representation, the input is already a *meaningful*
  token stream with a 30–172 symbol vocabulary. A byte-level learned tokeniser
  solves a problem that the grammar has already solved. **The default should be a
  plain embedding table over the prefix vocabulary**, with GBST retained as an
  ablation arm, not the trunk.
- If GBST is kept as an arm, implement it correctly: block-pooled candidates
  (mean over `b` consecutive positions), position-shared scores within a block,
  and mean-pool downsampling.

### 4.4 Attention
`XSA` currently means: post-softmax, remove the component of the attention output
along that position's own value vector. The stated motivation ("a token should not
trivially attend to itself") is a statement about the *weights*, implemented by
masking the diagonal of $QK^\top$ before softmax. These are different operators.
**Both must exist as named, separate arms** — `xsa_proj`, `xsa_mask`, `vanilla` —
and the claim tested against the right one.

### 4.5 FFN / MoE
Keep MoE, but:
- add the standard load-balancing auxiliary loss (importance + load, Shazeer et al.);
- log per-layer expert utilisation and routing entropy every epoch;
- replace the `for k in top_k: for e in n_experts:` double loop with a single
  scatter/gather dispatch (see `02` §3.3);
- gate the whole thing behind an ablation flag so `dense_ffn` is a first-class arm.

Without the balancing loss and the utilisation log, the "experts specialise by
token class" claim cannot be made at all.

### 4.6 Decoder
- Role-Filler embedding, tied output projection — keep (see `03` §4.3 on the name).
- Cross-attention over `[graph_enc | math_enc]` with modality embeddings — keep.
- **Gated fusion** as an arm: $g=\sigma(W[\,h_g;h_m\,])$ applied position-wise,
  so the decoder can weight topology vs algebra per position, and $g$ becomes a
  readable diagnostic.
- **Typed / constrained decoding** (§S8) — new, and the single cheapest accuracy win.

### 4.7 Capacity
v3 is 129.9M parameters against 324 (QED) / 210 (QCD) training examples —
~400k parameters per example. A capacity sweep
(`d_model ∈ {128, 256, 512}`, `layers ∈ {2, 4}`) is a required experiment, not an
optional one. The prior is that the small model wins.

---

## §5 Splits

Three split protocols, all reported. This is the part that decides whether the
final number means anything.

| protocol | construction | what it measures |
|---|---|---|
| **A — record** | random 80/10/10 over records, seeded | interpolation; SYMBA-comparable; the "validation accuracy" of the stated objective |
| **B — template** | grouped split on the 30 / 11 template classes; whole classes held out | genuine generalisation to unseen physics |
| **C — cross-theory** | train QED → test QCD (and reverse) | transfer of structure across gauge groups |

Protocol A is the headline number the objective asks for. Protocol B is the number
that says whether the model learned physics or a codebook. They will differ
enormously; that gap is a *result*, not an embarrassment.

**Train / val / test must be three disjoint sets.** The current code uses the test
set as the validation set for early stopping, checkpointing, and reporting. With
360 records an 80/10/10 split gives 36 val and 36 test — small, so all reporting
carries a Wilson interval, and protocol B additionally carries a
leave-one-class-out sweep so the effective n is the number of classes.

---

## §6 Evaluation

### 6.1 Metrics (all reported, always with n and 95% Wilson CI)
1. **Raw-string exact match** — comparable to SYMBA and to your own earlier runs.
2. **Symbolic equivalence** — `simplify(pred − gt) == 0` via sympy. The honest
   metric. Recall that 19 QCD raw pairs are algebraically identical, so (2) ≥ (1).
3. **Parse validity rate** — fraction of generated sequences that are a
   well-formed prefix expression at all.
4. **Mass-dimension validity** — fraction whose numerator is homogeneous of
   degree 4. Should be 100% under constrained decoding; under free decoding it is
   a sharp diagnostic.
5. **Denominator/channel accuracy** — did it pick the right propagator channel?
6. **Numerator monomial-set F1** and **coefficient exact-match rate** — decomposes
   "wrong answer" into "wrong structure" vs "wrong number".
7. **Template-class accuracy** — decode, canonicalise, map to leg-role form, and
   check which of the 30 / 11 classes was emitted.

Metrics 5–7 are what turn a single opaque percentage into a diagnosis. They are
the substance of the proposal's "diagnose and evaluate more professionally".

### 6.2 Decoding
Free-running only for headline numbers. Greedy and beam (width 4 and 8) reported
separately. **Teacher-forced accuracy may be logged but never reported as
"exact match"** — this mislabelling is the single most misleading thing in the
current results.

### 6.3 Mandatory baselines
Every headline number is reported next to: most-frequent template; exact lookup;
1-NN char-ngram retrieval (40.1% / 12.6% measured); and a **template oracle**
(accuracy achievable if the class were known and only the substitution had to be
predicted). Any architectural claim must clear the retrieval baseline by more than
the CI width.

### 6.4 Model selection
Not cross-entropy. With $\varepsilon=0.1$ smoothing the loss floor is 0.8778 and
the QED run reached 0.8828 — the selection signal was 5×10⁻³ wide and noise-
dominated. Select on **symbolic exact match on the validation split**, with loss
logged as a secondary trace.

---

## §7 Gate table — the tests that must pass between stages

Nothing downstream runs until the gate above it is green. This is the mechanism
that would have caught the wiring bug on day one.

| gate | between | assertion |
|---|---|---|
| G1 | S1→S2 | parsed records == non-blank lines; every record has 4 legs |
| G2 | S2→S3 | `standardize` is idempotent; no `\` without `%`; no digit-suffixed `%`-index survives |
| G3 | S3→S4 | `simplify(canonical − parse(raw)) == 0` for **every** record |
| G4 | S3 | numerator mass-dimension homogeneous, degree 4, for every record |
| G5 | S4→S5 | AST round-trip: `prefix → tree → infix` is symbolically equal to input; 0 failures |
| G6 | S4 | graph edge list is connected; every `OffShell` propagator resolves to two **distinct** vertices |
| G7 | S5→S6 | **no token id in val/test is absent from the train vocab** (or is counted and reported as OOV) |
| G8 | S6 | vocabulary built strictly after the split — asserted by construction order test |
| G9 | S6→S7 | **the tensor fed to the model is derived from the canonical/AST fields.** Explicit test: perturb `record['amp']` only, assert the model input is unchanged; perturb `record['amp_ast']`, assert it changes. *This is the test that the current code fails.* |
| G10 | S6→S7 | 0% of records truncated at every budget; padding mask sums equal true lengths |
| G11 | S7 | ablation flags are effective: `use_graph=False` must change the loss; parameter-count and grad-norm sanity per arm |
| G12 | S7 | train/val/test index sets are pairwise disjoint — asserted at every run start |
| G13 | S8→S9 | decode∘encode is identity on ground-truth targets |
| G14 | S9 | metric self-test on a synthetic perfect predictor → 100%, and on a shuffled predictor → ≈baseline |

G9 and G12 are the two that matter most. They are three lines each and they
invalidate or validate every number the project will ever produce.

---

## §8 Repository layout

```
symba/
  config/            single YAML per experiment; seed, split protocol, arms
  data/
    load.py          S1  parse + validate
    normalize.py     S2
    canonical.py     S3  sympy canonicalisation + mass-dimension check
    ast_parse.py     S4a Lark grammar (moved out of the notebook, unchanged)
    graph.py         S4b Feynman graph → edge list  (bug-fixed)
    serialize.py     S5  prefix + type tags
    vocab.py         S6  train-split-only vocabulary
    dataset.py       tensors, dynamic padding, length buckets
  model/
    embed.py         role-filler, type embeddings
    gbst.py          ablation arm only
    attention.py     vanilla | xsa_proj | xsa_mask
    ffn.py           dense | moe (+ load-balancing loss, utilisation logging)
    encoder.py  decoder.py  model.py
  train/
    loop.py  schedule.py  early_stop.py (on symbolic EM)
    diagnostics.py   per-type loss, expert utilisation, routing entropy,
                     cross-attention mass, probe hooks
  eval/
    decode.py        greedy | beam | constrained
    metrics.py       the seven metrics of §6.1
    baselines.py     majority | lookup | 1-NN | template oracle
    report.py        tables with n and Wilson CI
  tests/             the fourteen gates of §7
  notebooks/         thin drivers only — no logic
```

The rule that makes this stick: **notebooks import, they do not define.** Every
function currently defined in a cell moves into a module with a test.

---

## §9 Build order

1. S1–S3 + gates G1–G4. Canonicalisation verified on all 594 records.
2. S4–S6 + G5–G8. Grammar and graph moved out, round-trip proven.
3. Dataset + G9, G10, G12. **The wiring test comes before any model code.**
4. Baselines (§6.3) and the metric suite (§6.1) — before training anything, so the
   bar to clear is known in advance.
5. Smallest viable model (d_model 128, 2+2+2 layers) end-to-end on QED. Target:
   beat 40.1% 1-NN by more than the CI. If it cannot, the pipeline is wrong, not
   the capacity.
6. Capacity sweep, then arms: attention variant, FFN variant, fusion variant,
   modality ablation — one factor at a time, 3 seeds each.
7. Protocol B and C splits. Report the interpolation/generalisation gap.
8. Diagnostics and interpretability (see `03_proposed_changes.md` §5–§6).

---

## §10 Open questions to settle before step 1

1. **Is the canonical form the target, or the MARTY string?** Recommendation: train
   on canonical, report both. Needs an explicit decision because it changes what
   "exact match" means and how it compares to the SYMBA paper.
2. **Split unit for the headline number** — record (protocol A, matches the stated
   objective and prior work) or template (protocol B, honest). Recommendation:
   headline A, but never publish A without B beside it.
3. **QCD amplitude handling** — full 3072 context, per-diagram, or canonicalised
   amp. Needs one measurement (does `cancel` compress the amp as it compressed the
   target?) before deciding.
4. **Does the interaction+vertices pair determine the template class?** A quick
   lossy signature left 20–62% determinacy, which reflects the crudeness of the
   signature rather than a real ambiguity — the map is deterministic by physics.
   Establishing the *minimal sufficient* signature is a one-day experiment and it
   tells us exactly how much the graph pathway can possibly contribute.
