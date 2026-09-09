# 00 — Orientation: what this repo is, and where every number comes from

Written to answer one question: *if I open this repo cold, what am I looking at,
and which file produced which number?* Nothing here is new work. It is a map of
the code in this repository.

---

## 1. The one-line frame

The **architecture is unchanged** from the notebooks: dual-pathway encoder
(graph + math), cross-attention decoder over a fused memory, XSA / MoE /
role-filler as switchable arms.

What changed is **not the model**. It is three other things:

1. **What is fed in** — the AST and the graph object, rather than the raw
   `vertices` / `amp` strings the notebook was accidentally passing.
2. **What is asked for** — a canonicalised rational function, not MARTY's
   particular string, and not the first 65 bytes of it.
3. **How the answer is scored** — free-running decode, on a test split disjoint
   from validation, at one selected checkpoint, with an interval.

The repo *feels* foreign because it is now organised around (3). A notebook is
organised as a linear pipeline; this is organised as a measurement apparatus with
the pipeline as one of its parts. That is the whole of the disorientation.

---

## 2. Module map — what owns what

```
symba/
  config.py          every knob. A run == this dataclass tree + a seed.
  experiment.py      ARMS dict (name -> single-factor override), Job, run_arm
  data/
    load.py          read the MARTY dumps into Record objects
    normalize.py     index normalisation (_123456 -> stable slots)
    ast_parse.py     amplitude string -> AST
    graph.py         vertices -> Feynman graph (nodes, legs, propagators)
    canonical.py     TARGET DEFINITION + mass-dimension invariant + template_key
    splits.py        PROTOCOL A / PROTOCOL B
    serialize.py     expression <-> typed prefix token stream
    vocab.py         31-ish symbol typed vocabulary
    pipeline.py      build() -> the bundle everything else consumes
  model/
    embed.py         token + type + positional embedding, role-filler / TPR
    attention.py     vanilla | xsa_proj | xsa_mask
    ffn.py           dense | moe
    model.py         AmplitudeModel — assembles the above
  train/loop.py      training, free-running validation, checkpoint selection
  eval/
    decode.py        beam search + ConstraintMask (grammar-constrained decoding)
    metrics.py       symbolic EM, sequence EM, parse validity, dim-4, channel,
                     monomial F1, and the structure x coefficient decomposition
    baselines.py     most-frequent, 1-NN char n-gram, template oracle
```

Two files carry all the conceptual weight: **`data/canonical.py`** (what the
answer *is*) and **`data/splits.py`** (what the question *is*). Everything else
is machinery.

---

## 3. Template classes — the object both protocols are built on

Define, for a record with canonical target $f$ and leg-mass order
$(\mu_1,\dots,\mu_k)$, the substitution

$$\varphi: \mu_i \mapsto M_i \qquad (\text{leg role, not flavour})$$

and the class key

$$\kappa(f) = \mathrm{srepr}\big(\mathrm{cancel}(\varphi(f))\big).$$

Two records are in the same **template class** iff $\kappa$ agrees. This is an
equivalence relation on the corpus, and it collapses:

| theory | records | template classes | mean class size |
|---|---|---|---|
| QED | 360 | 30 | 12.0 |
| QCD | 234 | 11 | 21.3 |

So the corpus contains 360 QED *pairs* but only **30 distinct functional forms**.
The other 330 differ from one of those 30 only in which physical mass is
substituted into which leg slot.

That single fact is what makes two split protocols necessary.

---

## 4. Protocol A vs Protocol B

### Protocol A — record split (`split_protocol="record"`)

I.i.d. split at the **record** level. For each record independently,

$$h = \mathrm{SHA256}(\text{seed} \,\|\, \text{amp} \,\|\, \text{sq\_amp}) / 2^{64} \in [0,1),$$

$$h < 0.1 \Rightarrow \text{test},\quad 0.1 \le h < 0.2 \Rightarrow \text{val},\quad \text{else train}.$$

Hashing content rather than shuffling a list makes the split independent of
filesystem traversal order — a real bug class in the notebooks.

**Consequence.** A class $c$ is absent from training with probability
$(1-0.8)^{|c|} = 0.2^{|c|}$. For QED's mean $|c| = 12$ that is $\sim 4\times10^{-9}$.
So with probability indistinguishable from 1, **every test record's functional
form was seen in training** — only the mass substitution is new.

Protocol A therefore measures: *given that you have seen this functional form,
can you route the right masses into the right slots and emit it exactly?*
That is a real skill, and it is what the GSoC objective literally asks for. It
is **not** evidence of learned physics.

### Protocol B — template split (`split_protocol="template"`)

Split at the **class** level. Classes are ranked by a stable hash and assigned
*by count* (not by threshold — with 11 QCD classes a threshold can empty a split
by luck), so train / val / test hold **disjoint sets of functional forms**.

Protocol B measures: *emit a functional form you have never seen.*

### The gap is the result

| | QED | QCD |
|---|---|---|
| Protocol A, symbolic EM (3-seed mean) | 84.6% | 96.5% |
| Protocol B, symbolic EM | **0.0%** | **0.0%** |
| Protocol B, parse validity | 98.8% | 100% |
| Protocol B, mass-dimension-4 validity | **0.0%** | 100% |

Every baseline is also 0% under B, including the template oracle — which is 0%
*by construction*, since there is no same-class training record to retrieve.

The QED row is the diagnostically interesting one: **98.8% syntactically
well-formed, 0% dimensionally consistent.** Asked for a formula it has never
seen, the model writes a grammatical expression that is physically impossible.
That is precisely the separation the mass-dimension invariant was added to
detect — "wrong arithmetic" vs "wrong physics" — and it says the current 84.6%
is interpolation inside a codebook, not amplitude computation.

**Neither number alone is the honest report. The pair is.**

---

## 5. Provenance of every headline number

| number | file | arm | seed | protocol | n | epochs |
|---|---|---|---|---|---|---|
| **QCD 100.0%** | `results/QCD_record_long.json` | `full_vanilla_dense` | 0 | A | **14** | 120 |
| QCD 92.9% | `results/QCD_record_seeds.json` | `full_vanilla_dense` | 1 | A | 28 | 120 |
| QCD 96.8% | `results/QCD_record_seeds.json` | `full_vanilla_dense` | 2 | A | 31 | 120 |
| QED 83.0% | `results/QED_record_arms.json` (+ headline log) | `full_vanilla_dense` | 0 | A | 47 | 120 |
| QED 80.6% | `results/QED_record_seeds.json` | `full_vanilla_dense` | 1 | A | 31 | 120 |
| QED 90.3% | `results/QED_record_seeds.json` | `full_vanilla_dense` | 2 | A | 31 | 120 |
| QED/QCD 0.0% | `results/QED_templateB_long.json`, `QCD_templateB_long.json` | `full_vanilla_dense` | 0 | B | 102 / 48 | 120 |

**The 100% is: QCD, protocol A, control arm, seed 0, 14 out of 14 test records.**
Wilson 95% interval [78%, 100%]. It is one of three seeds and it is not the
result. The reportable figure is the 3-seed mean, 96.5% (sd 2.9).

---

## 6. What "symbolic exact match" actually computes

Not string equality. For prediction $p$ and reference $r$, written as rational
functions $N_p/D_p$ and $N_r/D_r$:

$$\text{match} \iff \mathrm{expand}(N_p D_r - N_r D_p) = 0.$$

Cross-multiplication rather than `simplify(p - r) == 0` — exact for rational
functions and ~2 orders of magnitude cheaper, which matters because this runs on
every validation sample at every eval.

Two guards short-circuit it, and **both can only make the metric pessimistic**:

- `OPERATOR_SLACK = 3` — a prediction with more than 3× the reference's operator
  count is scored wrong without being parsed.
- `COMPLEXITY_CAP = 600` — a parsed expression above 600 sympy ops is scored
  wrong without being expanded.

Rejections are counted and reported as `complexity_bailouts`. This is safe
(a correct answer cannot be 3× more complex than the canonical target *in
canonical form*) but it is an assumption, not a theorem, and it deserves a
sentence in any write-up.

`COMPLEXITY_CAP` is also weaker than it looks. `count_ops` measures the size of
the *written* expression, not the cost of expanding it: a product of k binomials
has `count_ops` 11 for every k, while its expansion grows as $2^k$. The cap is a
correctness guard, not a cost guard. See `04_change_review.md` §5.

---

## 7. Open scrutiny points

Things that are currently *not* airtight, in order of how much they would move
the picture:

1. **The canonical target is not the literal task target.** The task says
   "squared amplitudes are the target sequences"; the model predicts
   `expand → together → cancel` of that string. Algebraically identical,
   materially easier (QCD worst case 2872 chars → 258). The `raw_target` arm
   exists to measure the cost of this and **has not been run**. This is the
   single most likely thing to draw a reviewer's fire.
2. **The 96.5% QCD mean is unweighted across seeds with $n = 14, 28, 31$.**
   Pooled it is $70/73 = 95.9\%$. Small difference, but the unweighted mean
   gives the smallest, luckiest test set equal weight.
3. ~~**`RESULTS.md` §"What is not yet measured" is stale.**~~ Fixed in the
   pipeline-correctness pass; see `04_change_review.md` D6. `RESULTS.md` now
   also carries a staleness banner, because three of the changes in that pass
   move the numbers it reports.
4. **QCD test $n = 14$ on seed 0** makes 100% and 93% statistically
   indistinguishable.
5. **Protocol B QCD is 6/2/3 classes** — a single arbitrary grouping.
   Leave-one-template-out is the right design and is not implemented.
6. **No QED arm separates from any other** (78.7–83.0, intervals ~22 points
   wide). Reported as a null result, which is correct, but it means the XSA and
   MoE contributions are currently unsupported at this corpus size.
