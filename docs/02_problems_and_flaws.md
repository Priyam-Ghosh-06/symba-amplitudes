# 02 — Problems and Flaws in the Current Code

Scope: everything wrong with the code **as written**, holding the intended
architecture fixed. Conceptual and architectural redesign is in
`03_proposed_changes.md`. Numbers referenced here are derived in
`01_architecture_design.md` §0.

Severity key: **[F]** fatal — invalidates results · **[H]** high — materially
distorts results · **[M]** medium — correctness or reproducibility ·
**[E]** efficiency · **[L]** latent — not biting on this data, will bite later.

---

## §1 The wiring — results-invalidating

### 1.1 **[F]** The model never sees the AST or the graph

`FeynmanDataset.__getitem__` (v3 Cell 10, baseline Cell 15):

```python
graph_str  = item.get('vertices', '')     # raw vertices string
math_str   = item.get('amp',      '')     # raw infix amplitude
target_str = item.get('sq_amp',   '')     # raw infix target
```

and the loaders are constructed from the **raw** lists:

```python
feynman_loaders = {"qed_train": make_feynman_loader(qed_train, ...), ...}
#                                                   ^^^^^^^^^ not clean_qed_train
```

`clean_dataset()` returns a *new* list; `apply_math_ast()` and `process_graphs()`
mutate `clean_*` in place. Nothing they produce is ever read.

Never reaches the model: `perfect_standardizer` (dummy-index normalisation),
`clean_graph_topology`, the Lark grammar → `amp_ast` / `sq_amp_ast`,
`FeynmanGraph` → `graph_flat`, `QFTCharDataset`, `dynamic_pad_collate`.

What runs is a byte-level character transformer over two raw infix strings with
`%\sigma_18923`-style indices intact. Every claim resting on algebraic hierarchy
or interaction topology is untested — not disproved, untested.

**Fix:** feed `graph_flat`, `amp_ast`, `sq_amp_ast`; build loaders from `clean_*`;
add gate G9 (perturb `record['amp']` → model input must be unchanged; perturb
`record['amp_ast']` → it must change).

### 1.2 **[F]** 100% of inputs and targets are truncated

`MAX_TARGET_LEN = 65` bytes against a QED median of 127 and a QCD median of 421
(QCD p90 = 2409, max = 2872). `MAX_MATH_LEN = 168` against QED median 246, QCD
median 650, max 4734.

The reported exact match is exact match on **the first ~63 characters** — roughly
half a median QED target, roughly 15% of a median QCD target — from ~2/3 of a QED
amplitude and ~25% of a QCD one.

**Fix:** budgets 192 / 192 / 320 (QED amp) / 3072 (QCD amp) on the canonical
prefix representation, which covers 100% (`01` §0.5); dynamic padding; truncation
raises rather than silently clipping.

### 1.3 **[F]** Two parallel data pipelines, the better one dead

`QFTCharDataset` + `dynamic_pad_collate` (length-sorted, pads to a multiple of 12
so GBST's block sizes divide evenly) is fully built, validated with shape
assertions, printed as `✅ All GBST DataLoaders ready` — and never used.
`FeynmanDataset` with fixed 120/168/65 padding replaces it silently.

Two pipelines that disagree is how 1.1 and 1.2 survived review. **Fix:** delete
one. P1 in `01` §1.

### 1.4 **[H]** The symbolic metric cannot fire

```python
re.sub(r'(%[a-zA-Z\\]+_)(d\d+)', replacer, expression)
```

requires indices already in `d1, d2, …` form. The model's data was never
standardised (1.1), so the pattern matches nothing and

```
symbolic_exact == beam_exact
```

**identically, by construction** — in both notebooks, both theories. The equality
in the output is not the model getting equivalence classes right; it is a no-op.

### 1.5 **[H]** Test set used as validation set

```python
val_loader = feynman_loaders[test_key]
```

Early stopping, best-checkpoint selection, and the reported metric all read the
same 36 (QED) / 24 (QCD) records. There is no held-out set anywhere in the project.

### 1.6 **[H]** Reported metric is a maximum over epochs

`plot_ablation_bars` and `plot_xsa_vs_sa` both use `max(hist) * 100`. QED's
94.44% (34/36) occurred on an epoch logged `no improve 1/5` — i.e. not the
checkpoint that was saved. Maximum-over-epochs of a test statistic on n=36 is
a biased estimator; with ~25 epochs of noise the upward bias is several points.

**Fix:** report the metric at the selected checkpoint, chosen on a disjoint
validation split.

### 1.7 **[H]** "Exact match" and "greedy exact" are teacher-forced

`evaluate_complexity_vs_error` feeds the ground-truth target as decoder input and
takes `argmax`. So the per-epoch `exact match` in every training log is a
teacher-forced statistic. In the beam-search cell, `greedy_exact` does the same:

```python
logits, _ = model(graph_t…, math_t…, target_t[:-1]…)   # ground truth as input
greedy_str = decode_bytes_to_str(logits[0].argmax(dim=-1).tolist())
```

Only `beam_exact` is genuinely free-running. The QED "88.9% greedy vs 94.4% beam"
comparison is therefore not greedy-vs-beam; it is teacher-forced-vs-beam.

---

## §2 Data handling

### 2.1 **[M]** The split is not reproducible

`build_structured_datasets` iterates `os.listdir(data_directory)`, whose order is
filesystem-dependent. `train_test_split(..., random_state=42)` then permutes a
list whose *order differs between machines*. The split is deterministic on one
machine and different on another. **Fix:** `sorted(os.listdir(...))`, and hash the
record (e.g. the `amp` string) to assign the split so it is order-independent.

### 2.2 **[M]** No global seeding

`torch`, `numpy`, and `random` are never seeded. Weight init, dropout, MoE
routing noise, and DataLoader shuffling are all unseeded. No run in this project
is reproducible even on the same machine.

### 2.3 **[M]** Vocabulary leaks across the split (`Tokenize_final` Cell 12)

The vocabulary is built by counting tokens over **all** equations, and the 80/10/10
split happens two cells later (Cell 14). The proposal states "vocabulary built
from the training split only" — the code does the opposite.

Additionally the id assignment iterates `token_counts.items()`, i.e. dict
insertion order, which depends on corpus traversal order — ids are not stable
across data orderings. **Fix:** sort by `(-count, token)`.

### 2.4 **[M]** `FeynmanGraph` produces self-loops instead of propagators

```python
prop_match = re.search(r'(.*?)\((V_\d+)\)', p)
if prop_match and "OffShell" in p:
    self.propagators.append((v_id, prop_match.group(2), ...))
```

For `V_1: … OffShell A(V_1)` this yields `('V_1','V_1','OffShell A')` — a self-loop.
The validation output in the notebook shows exactly this:

```
Propagators : [('V_1','V_1','OffShell A'), ('V_0','V_0','OffShell A')]
```

The internal line joining $V_0$ and $V_1$ is never represented. The "graph"
pathway therefore encodes two disconnected vertex bags plus in/out legs, with no
edge between them — the propagator, which is precisely what determines the
denominator channel, is absent.

**Fix:** resolve `OffShell X(V_i)` by matching the propagator particle type across
vertices and emitting `(V_i, V_j, X)` with $i \neq j$; assert graph connectivity
(gate G6).

Note also that `flatten_for_transformer` iterates `self.nodes.items()` — insertion
order, i.e. the order vertices happen to appear in the string. There is no
canonical vertex ordering, so two identical diagrams written in different orders
produce different token sequences.

### 2.5 **[M]** The two halves of the project disagree on the data format

Model notebooks: `line.split(" : ")`, 4 fields (interaction, vertices, amp, sq_amp).
Tokeniser notebooks: `text.rsplit(":", 2)`, 3 fields — so their `interaction`
variable silently contains *interaction + vertices concatenated*. The tokeniser was
designed and validated against a different parse of the same file.

### 2.6 **[M]** Pair-level split is claimed but not implemented

`Tokenize_final` Cell 4 appends `amp` and `sq_amp` as **separate entries** of one
flat list, then Cell 14 shuffles that flat list and slices 80/10/10. An amplitude
and its own squared amplitude routinely land in different partitions. The proposal
states the split was made at tuple level to prevent exactly this.

### 2.7 **[L]** Aliasing in the split initialisation

```python
qed_train = qed_test = qcd_train = qcd_test = []
```

binds all four names to **one** list object. Harmless only because every branch
reassigns; if any dataset were empty and something appended, all four would mutate.

### 2.8 **[L]** Byte encoding takes only the first byte of a multi-byte character

```python
ids.append(char.encode('utf-8')[0] + self.BYTE_OFFSET)
```

`GBSTByteTokenizer.encode` iterates *characters* and keeps byte 0 of each. Any
non-ASCII character is silently corrupted to its lead byte. The corpus is ASCII
today, so this is latent — but `\tau`, `\nu` appear as escaped ASCII and a future
EW dataset with real Unicode would break silently. `_to_byte_tensor` in
`FeynmanDataset` does it correctly (`text.encode('utf-8')`); the two disagree.

### 2.9 **[L]** `drop_last=True` on the training loader

QED: 324 items, batch 16 → 20 batches = 320, so 4 records are dropped per epoch
(re-shuffled each epoch, so all are seen eventually). QCD: 210 → 208, 2 dropped.
With 324 examples this is 1.2% of the data per epoch for no benefit.

---

## §3 Model-level bugs and waste

### 3.1 **[H]** No padding masks anywhere

Neither `XSAFlashAttention` nor either attention in `DecoderBlock` receives an
`attn_mask` or `key_padding_mask`. `F.scaled_dot_product_attention` is called with
`q, k, v` only. Every PAD position participates as both query and key, in the
encoders and in the cross-attention memory.

With fixed-length padding to 120/168/65 and true lengths well below the caps for
`vertices`, a substantial share of the memory the decoder cross-attends to is
padding. The PAD embedding has `padding_idx=0` so it starts at zero, but the
positional encoding is added afterwards and the convolutions and LayerNorm make
it non-zero immediately.

**Fix:** build masks in the collate function; pass them through GBST (mask before
convolution), encoders, and cross-attention.

### 3.2 **[M]** GBST is not GBST

Charformer forms candidate subwords by **pooling `b` consecutive positions** into
blocks, scores each block, and shares the score across positions inside a block —
the segmentation is the point. Here each block size is

```python
nn.Conv1d(d_model, d_model, kernel_size=b, padding='same')
```

producing a *same-length* candidate scored independently per position. That is a
learned multi-scale local mixer. It may work fine; it is not the cited method, and
the proposal's justification ("the model learns physically meaningful token
boundaries") does not describe what the code does — there are no boundaries.

Secondary: `padding='same'` with even kernels (2 and 4) is asymmetric and triggers
the PyTorch warning visible in the notebook output; the receptive field is
off-centre for half the block sizes.

### 3.3 **[E]** MoE dispatch is O(top_k × n_experts) masked passes

```python
for k in range(self.top_k):            # 2
    for e_idx in range(self.n_experts):  # 4
        mask = (expert_ids == e_idx)
        output[mask] += gate_k[mask] * self.experts[e_idx](x_flat[mask])
```

Eight boolean-mask gathers per FFN call, each materialising a copy of the selected
rows, and eight separate expert invocations. With 12 MoE blocks (4 encoder graph +
4 encoder math + 4 decoder) that is 96 masked expert calls per forward pass.

**Fix:** one `argsort` on the flattened expert assignment, one `index_select`, a
single grouped pass per expert (each expert called exactly once), then
`index_add_` back. Reduces to `n_experts` calls and removes the inner loop.

### 3.4 **[H]** MoE has no load-balancing loss and no utilisation logging

Noisy top-2-of-4 gating with nothing penalising router collapse and nothing
measuring it. The proposal's Claim 3 ("routing distinct token classes — gamma
matrices vs kinematics — to specialised sub-networks") has no supporting
measurement anywhere in the code. It may be true; nothing in the repository could
tell you either way.

**Fix:** add the standard importance + load auxiliary loss; log per-layer expert
utilisation histogram and routing entropy every epoch; compute mutual information
between token type and expert assignment.

Minor semantics: `gates = softmax(top_scores)` normalises over the selected top-k
only. GShard/Switch softmax over all experts then renormalise the selected subset.
The difference changes the gradient the router receives for unselected experts.

### 3.5 **[M]** Role-Filler is an over-parameterised linear map

```python
return self.norm(self.C_e(self.A_e(x) + self.B_e(role)))
```

with `A_e`, `B_e` linear-without-bias and `C_e` linear-with-bias, and **no
nonlinearity between them**. Therefore

$$C_e\big(A_e x + B_e r\big) = (C_e A_e)\,x + (C_e B_e)\,r + b$$

which is exactly representable by two linear maps. `A_e` and `B_e` are redundant
with `C_e`: the module carries $3 d^2$ parameters where $2 d^2$ suffices, and the
extra factorisation adds no expressivity — only a worse-conditioned optimisation
landscape (a product of two unconstrained matrices).

At `d_model=512` that is 262k wasted parameters per instance × 3 instances
(graph projection, math projection, decoder embedding) ≈ 0.8M.

**Fix (keeping the architecture):** either drop `A_e`/`B_e`, or insert a
nonlinearity so the factorisation means something. See `03` §4.3 for the deeper
point about the name.

### 3.6 **[M]** Tied embeddings do not tie what the argument requires

```python
self.fc_out.weight = self.role_filler_embed.filler_embed.weight
```

The decoder input is not `filler_embed(x)` — it is
`norm(C_e(A_e(filler_embed(x)) + B_e(role)))`. Two learned linear maps and a
LayerNorm sit between the tied matrix and the representation the decoder actually
uses. The geometric justification for weight tying (input and output token
embeddings live in the same space, so the output logit is an inner product with
the token's own embedding) does not hold here. Also `fc_out` retains a bias, and
`_init_weights` writes into the shared tensor twice via `apply`.

### 3.7 **[M]** Initialisation is fan-in blind and skips the convolutions

`_init_weights` sets `normal_(0, 0.02)` for every `nn.Linear` and `nn.Embedding`
regardless of shape. For `d_model=512`, Xavier would be $\approx 1/\sqrt{512}=0.044$;
for the `512→2048` expert projections it is smaller still. Consistently
under-initialised weights plus pre-LN gives a very small effective residual branch
at step 0 — recoverable, but it is a choice with no stated derivation.

`nn.Conv1d` is not matched by the isinstance check, so the four GBST block convs,
the four score projections, and the downsampler keep PyTorch defaults. The model
has two different initialisation schemes by accident.

### 3.8 **[M]** Declared dropout is almost entirely absent

`DROPOUT = 0.2` is applied once, to the GBST output. There is no dropout in
attention, in the FFN, or on the residual branches, anywhere in either notebook.
With 129.9M parameters and 324 training examples this is the single largest
missing regulariser.

### 3.9 **[M]** Baseline shares one positional-encoding *instance* across both encoders

```python
self.pos_encoder = HybridPositionalEncoding(d_model)   # one instance
... g = self.pos_encoder(graph_x) ... m = self.pos_encoder(math_x)
```

The learned positional table is shared between the graph and math streams, which
directly undercuts the baseline's stated purpose ("dual-pathway prevents early
modality interference"). v3 fixes this with two `RoleFillerProjection` instances —
so the baseline-vs-v3 comparison silently confounds this fix with Role-Filler and
MoE.

Separately: `HybridPositionalEncoding` adds a fixed sinusoidal term of order 1 to a
learned table initialised at `N(0, 0.02)`. Early in training the learned component
contributes ~2% of the positional signal, so "hybrid" is nearly vacuous for the
first several epochs.

### 3.10 **[E]** `MAX_SEQ_LEN = 2048` positional tables for sequences ≤ 168

`nn.Embedding(2048, 512)` = 1.05M parameters per table, in
`HybridPositionalEncoding`, `RoleFillerEmbedding`, and both
`RoleFillerProjection`s. At most 168 rows are ever indexed. ~3–4M parameters are
allocated, initialised, weight-decayed, and never touched by a gradient.

### 3.11 **[L]** XSA epsilon is absolute, not relative

```python
Y = Y - (dot_YV / (norm_V_sq + 1e-6)) * v
```

`norm_V_sq` is $\|v_i\|^2$ over `d_head=64` dimensions. With `normal_(0,0.02)`
initialisation the value vectors start small, so at step 0 the additive `1e-6`
is not negligible relative to $\|v\|^2$ and the projection is systematically
under-applied. Use a relative epsilon or clamp.

### 3.12 **[L]** Ablation flags are ignored by beam search

`beam_search_decode` calls `model.encoders(g_emb, m_emb)` with default arguments —
`use_graph=True, use_math=True, use_xsa=True`. Any attempt to beam-evaluate the
`graph_only`, `math_only`, or `full_sa` arms silently evaluates the full XSA model
with the wrong weights. Only `full_xsa` was beam-evaluated, so no published number
is affected, but the ablation table cannot be extended to free-running decoding
without hitting this.

---

## §4 Decoding

### 4.1 **[H]** Beam search has no length normalisation

Beams are ranked by raw cumulative log-probability:

```python
beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
```

Every additional token adds a negative log-probability, so a beam that emits EOS
early always outranks a longer beam that is still correct. This systematically
biases generation toward short outputs — which, in a regime where the target is
already truncated to 63 bytes, is exactly the failure mode that would be invisible.

**Fix:** normalise by length, or use the GNMT length penalty
$\ell(Y)=\frac{(5+|Y|)^\alpha}{6^\alpha}$ with $\alpha \in [0.6, 1.0]$, and report
the choice.

### 4.2 **[E]** Beam search has no KV cache and re-encodes the full prefix each step

```python
for _ in range(max_len - 1):
    for seq, score in beams:
        decoder_input = torch.tensor([seq], ...)     # whole prefix, every step
        x = model.decoder.role_filler_embed(decoder_input)
        for layer in model.decoder.layers: x, _ = layer(x, memory, ...)
        logits = model.decoder.fc_out(x)
        last_logits = logits[0, -1, :]               # 99% of the compute discarded
```

Cost is $O(W \cdot T^2)$ in attention work where $O(T)$ suffices with a cache. For
$W=4$, $T=64$: ~256 decoder passes per sample over prefixes of mean length 32,
with all but the last position's logits thrown away. Beams are also processed one
at a time rather than batched, and the whole evaluation runs at batch size 1.

**Fix:** batch the beams into the batch dimension (`W` rows), cache self-attention
K/V, and cache the cross-attention K/V once per sample (the memory is fixed).
Expected speedup 30–100×, which matters once the target budget goes from 65 to 192.

### 4.3 **[M]** No constrained decoding

Nothing prevents the model emitting a sequence that is not a well-formed
expression. Given that (a) the grammar is known, (b) the target vocabulary is 30
symbols, and (c) the numerator is mass-dimension-4 homogeneous in 100% of records,
a typed prefix decoder could make ill-formed output *impossible*. This is the
cheapest available accuracy gain and it is entirely absent. See `03` §3.

---

## §5 Training loop and instrumentation

### 5.1 **[H]** Model selection uses a saturated objective

With $\varepsilon=0.1$ and $V=260$ the cross-entropy floor is

$$H(q) = -\Big[q_1\log q_1 + (V-1)\,q_0\log q_0\Big] = 0.8778,\quad q_1 = 1-\varepsilon+\tfrac{\varepsilon}{V},\ q_0=\tfrac{\varepsilon}{V}$$

Best QED val loss: **0.8828** (Δ = 0.0050). Best QCD: **0.9083** (Δ = 0.0305).
`patience=5` was therefore comparing differences in the fourth decimal of a
saturated loss while free-running exact match was still climbing 75% → 94%. The
selection signal and the reported metric had already decoupled.

Also: label smoothing on a **deterministic** symbolic map deliberately prevents the
model from being confident about a mapping that is exactly deterministic. That is
a defensible regulariser for a small dataset, but it must be a stated choice with
an ablation, not a default.

Secondary: smoothing distributes probability mass onto class 0 (`PAD`), which can
never be a legitimate target since `ignore_index=0`.

### 5.2 **[E]** Two full validation passes per epoch

`validate_epoch` and `evaluate_complexity_vs_error` each run a complete forward
pass over the validation loader every epoch, for the same batches, differing only
in whether they compute the loss or the argmax. Merge them.

### 5.3 **[E]** History stores per-sample results for every epoch

`history['complexity_results'].append(complexity_results)` accumulates a list of
36 dicts per epoch across 30 epochs × 8 runs, and only `hist[-1]` is ever read.

### 5.4 **[M]** Heatmaps are computed on the training set

```python
train_key = f"{physics.lower()}_train"
loader    = feynman_loaders.get(train_key)
heatmap   = get_graph_heatmap(model, loader, device, num_samples=4)
```

`plot_heatmaps` reads the **train** loader. Test 2's interpretability figures are
attention over training data — memorised examples. Combined with a sample size of
4 and no aggregation, "no consistent interpretable patterns" is an unsurprising and
uninformative result.

### 5.5 **[M]** The plotted LR schedule is not the schedule used

```python
steps_per_epoch = 100          # approximate; adjust to your actual loader size
```

Actual values are 20 (QED) and 13 (QCD). Figure 08 shows a schedule with 3000
steps and 300 warmup; the runs used 600 total and 60 warmup. Purely cosmetic, but
it is a published figure that does not describe the experiment.

### 5.6 **[M]** `lr_lambda(0) = 0`

`return step / max(1, warmup_steps)` at `step=0` gives an exactly zero learning
rate for the first optimiser step. Harmless, but the first step is wasted and the
sanity-check printout (`LR after step 1: 0.0000030`) is a symptom, not a feature.

### 5.7 **[M]** Complexity features are computed on a different object than the error

`_extract_metadata` counts brackets, characters, and operators on the **full raw**
`sq_amp`; `token_error` is measured on the **truncated 63-byte** target. Test 3
correlates a whole-expression statistic against a prefix error rate. "Limited
variation, no clear trend" is at least partly an artefact of this mismatch, not a
statement about the physics.

Additionally, `depth` counts bracket nesting on infix text — a quantity that
becomes meaningless once the representation is prefix notation with brackets
eliminated, which was the entire point of the AST work.

### 5.8 **[M]** The baseline's results are not in the baseline notebook

`Sq_amp_calculation(baseline)` Cell 16 terminates in `KeyboardInterrupt` after two
epochs (`token acc 27.26%, exact match 0.00%`). The figures in Cell 17 were
produced from an `all_histories` that exists only in a dead kernel. The proposal's
"baseline QED ~70%, QCD ~40%" cannot be reproduced or checked from any saved
artefact in the repository.

### 5.9 **[M]** No per-position, per-type, or per-class error breakdown

The only training-time signals are a scalar loss, a scalar token accuracy, and a
scalar exact match. There is no way to tell whether failures are early or late in
the sequence, in coefficients or in structure, or concentrated in particular
template classes. Everything in `03` §5 is unavailable because none of it is
recorded.

---

## §6 Silent-failure patterns to eliminate

These are the *habits*, not individual bugs, and they are what allowed §1 to
survive:

1. **Bare `except Exception` with a fallback value.**
   ```python
   except Exception:
       item["amp_ast"] = [UNK_TOKEN]; item["sq_amp_ast"] = [UNK_TOKEN]
   ```
   A parse failure becomes a valid-looking record. Same in `process_graphs`
   (`graph_flat = ["<UNK>"]`) and in `decode_bytes_to_str`
   (`except: return ""`, plus `errors='replace'`).
   **Rule:** a parse failure is a build failure; log the offending pattern and stop.

2. **`.get(key, default)` on fields that must exist.** `item.get('amp_ast', [])`
   returns an empty list for a record where the field was never set — which is
   exactly the class of bug in §1.1, made invisible.
   **Rule:** `item['amp_ast']`, and let the `KeyError` happen.

3. **Assertions that check shapes but not semantics.** Every notebook cell ends
   with `assert x.shape == (...)` and a `✅`. Not one asserts *which field* the
   tensor came from. The shape tests all pass on the wrong data.
   **Rule:** at least one semantic assertion per stage (gate table, `01` §7).

4. **Validation on mock tensors.** `torch.randint(0, VOCAB_SIZE, (16, 336))`
   verifies that shapes propagate. It cannot detect that the real pipeline feeds a
   different field. Every module in both notebooks is validated this way.
   **Rule:** at least one end-to-end test on a real record.

5. **Notebook-state dependence.** Results, figures, and checkpoints depend on
   variables from cells that may have been run in a different order or in a
   previous session (`all_histories` in §5.8 is the concrete instance).
   **Rule:** logic in modules, notebooks as thin drivers.

---

## §7 Fix order

Ordered by (results-invalidating first, then cost of leaving it):

| # | item | § | effort |
|---|---|---|---|
| 1 | wire AST + graph into the dataset; delete the dead pipeline | 1.1, 1.3 | S |
| 2 | remove truncation; canonical representation; dynamic padding | 1.2 | M |
| 3 | three-way split; disjointness assertion; seed everything | 1.5, 2.1, 2.2 | S |
| 4 | report at checkpoint, free-running decoding only | 1.6, 1.7 | S |
| 5 | padding masks throughout | 3.1 | M |
| 6 | fix the propagator self-loop; canonical vertex ordering | 2.4 | S |
| 7 | model selection on symbolic EM, not smoothed CE | 5.1 | S |
| 8 | beam length normalisation + KV cache + batched beams | 4.1, 4.2 | M |
| 9 | MoE load-balancing loss + utilisation logging + scatter dispatch | 3.3, 3.4 | M |
| 10 | vocabulary from train split only; deterministic ids | 2.3 | S |
| 11 | dropout in attention/FFN; capacity sweep | 3.8 | S |
| 12 | trim positional tables; collapse the redundant Role-Filler linears | 3.5, 3.10 | S |
| 13 | constrained/typed decoding | 4.3 | M |
| 14 | diagnostics: per-position, per-type, per-class error | 5.9 | M |
| 15 | replace `except`-with-fallback everywhere; add the gate tests | §6 | M |

Items 1–4 alone change what every number in the project means. Nothing else should
be attempted before they are done and the gates in `01` §7 are green.
