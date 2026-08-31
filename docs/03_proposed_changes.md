# 03 — Proposed Changes

Conceptual, architectural and methodological changes to the framework, with the
objective held fixed: **maximise squared-amplitude accuracy on held-out QED and
QCD data.** `01` is the blueprint, `02` is the bug list; this document is the
argument for *why* the blueprint looks the way it does, and what to build beyond it.

---

## §1 The reframing

Applying the method you use on any research problem — state the raw system
honestly first, then ask what the prior work did to circumvent its ill-posedness,
then ask what our setting actually changes.

### 1.1 The raw system

The map we are learning is

$$\Phi:\ \big(\text{diagram }\mathcal{D},\ \text{amplitude }\mathcal{M}\big)\ \longmapsto\ \overline{|\mathcal{M}|^2}$$

$\Phi$ is **exactly deterministic**. It is not a translation, not a conditional
distribution, not ambiguous. It is the composition of squaring, spin/colour
summation with the completeness relations, Dirac-trace evaluation, and algebraic
simplification. There is exactly one right answer and it is computable.

That determinism is the whole difficulty, and it is where the sequence-modelling
framing quietly breaks. A seq2seq model with cross-entropy is a maximum-likelihood
estimator of $p(y\mid x)$ over a token alphabet. It is a good estimator when the
target has genuine entropy given the input. Here $H(Y\mid X)=0$. Every bit of
apparent difficulty comes from the *representation*, not the map: bracket nesting,
dummy-index labels, MARTY's unsimplified output, operator ordering. **The model is
being asked to learn a deterministic function through a lossy, redundant,
non-canonical encoding of it.** That is the raw ill-posedness — not of $\Phi$, but
of the encoding we hand the network.

### 1.2 What the base paper circumvents, and its price

SYMBA (Alnuqaydan et al.) circumvents this by declaring the problem to be
sequence-to-sequence and paying for it with scale and with a metric: it trains a
large transformer on a large corpus and reports *sequence accuracy* against
MARTY's output string. The price is threefold — data hunger (the structure has to
be inferred rather than given), brittleness (there is no guarantee the output is
even a well-formed expression), and a metric that punishes algebraically correct
answers written differently. Your proposal identifies the first of these correctly.
It does not yet name the second or the third.

### 1.3 What our setting actually changes

We have 360 QED and 234 QCD records — two orders of magnitude below the regime
where "let the transformer infer the structure" is a reasonable bet. So the
inductive bias is not a nice-to-have; it is the only lever. Good.

But we should be honest about what the data is. Measured (`01` §0.6): after
mapping each mass symbol to the leg index of its first occurrence and
canonicalising,

> **QED: 360 records → 30 distinct functions. QCD: 234 records → 11.**

The corpus is 30 and 11 *formulas*, instantiated with different flavour labels.
That single fact reorganises everything:

- The learning problem is not "translate 360 sequences". It is **(i) identify
  which of ~30 functional forms applies, and (ii) bind the right symbols into it.**
- Under a random record-level split, ≈100% of test records belong to a template
  seen in training. So the reported number measures *substitution within a seen
  template*, which is a real skill but a much smaller one than the proposal claims.
- The effective sample size for any generalisation claim is 30 and 11, not 360 and
  234. A 36-record test set drawn from 30 classes is not 36 independent trials.

This is not a reason for despair. It is the most useful thing we know, because it
tells us exactly what a good architecture should do, and it happens to coincide
with what a physicist does.

### 1.4 The spine: three things a physicist does that the model does not

Your stated interest is in taking simple, overlooked features of human learning as
architectural guidance. Here they are not metaphorical — each maps to a concrete,
measured, high-leverage change:

| a physicist… | the model currently… | the change |
|---|---|---|
| **simplifies before comparing** | compares MARTY's raw string byte-for-byte | canonicalise (§2.1) — collapses the QCD tail 11× and merges 19 wrongly-distinct targets |
| **learns one formula and substitutes flavours** | memorises 360 strings | leg-role abstraction + explicit binding (§2.2, §6.2) — 360 → 30 |
| **checks dimensions before believing a result** | has no notion of validity | mass-dimension-4 constraint, exact in 594/594 records (§3.2) |

These three are the argument of the project. Everything else is engineering.

---

## §2 Representation — the highest-leverage lever

### 2.1 Canonicalisation of the target

Measured (`01` §0.4): `expand → together → cancel` takes QCD's target from
median 421 / p90 2409 / max **2872** characters to median 205 / p90 247 / max
**258**. The length explosion that motivated the 65-byte cap does not exist in the
mathematics; it is MARTY emitting nested, unsimplified output. It also merges
118 → 99 distinct QCD targets, i.e. **19 raw strings are algebraically identical
to another raw string** — so the raw-string metric currently marks correct answers
wrong.

Serialised as prefix with digits split, the canonical target has a **30-symbol
vocabulary** and a **maximum of 125 (QED) / 143 (QCD) tokens**. One budget of 160
covers 100% of both corpora, against a 260-symbol byte alphabet truncating 100% of
targets at 63.

**This is the single change that most improves the achievable ceiling**, and it
costs one sympy call per record.

*The honest cost.* It changes what "the answer" is, from MARTY's particular string
to the mathematical object. Both must be reported (`01` §6.1): raw-string EM for
comparability with SYMBA and with your own earlier runs, symbolic-equivalence EM as
the metric you actually defend. Hiding this would be the same class of error as the
truncation.

### 2.2 Leg-role abstraction and the binding problem

The 360 → 30 collapse comes from a substitution: $m_e \to M_1$, $m_\mu \to M_3$,
by leg index. If the model predicted the *template* in leg-role variables and
separately predicted the *binding* $\{M_i \mapsto m_X\}$, it would be solving the
problem the way it is actually structured.

This has a strong empirical warrant beyond the class count. For QED, partition the
corpus by process type:

- **216 records** are four-fermion processes (two distinct charged flavours, no
  external photon). For **all 216**, the leading rational prefactor equals
  $$\tfrac14\,(Q_a Q_b)^2$$
  exactly — $\tfrac14$ for $e\mu$ ($Q=-1,-1$), $\tfrac{4}{81}$ for $u\,t_{\rm op}$
  ($Q=\tfrac23,\tfrac23$: $\tfrac14\cdot\tfrac{16}{81}$). Zero exceptions.
- **144 records** are Compton-like (one charged flavour + photon legs) and follow a
  different, equally derivable rule.

So the flavour dependence of the answer is not learned structure — it is an
explicit algebraic function of the leg quantum numbers. Two consequences:

1. **Exact augmentation.** Substituting a flavour and rescaling the prefactor by
   $(Q'_aQ'_b)^2/(Q_aQ_b)^2$ maps a valid pair to a valid pair, provably. This lets
   us synthesise training data with a physics guarantee rather than a heuristic.
   The corpus already contains many of these variants, which is *why* there are 30
   classes — but the augmentation extends beyond what MARTY generated, and it is
   the one lever that genuinely increases the number of independent examples.
2. **Leakage control.** Flavour-substituted variants of one template must not
   straddle train and test. Protocol B in `01` §5 exists for exactly this.

**Open question worth one day:** whether the Mandelstam identifications
$s_{12}\equiv s_{34}$, $s_{13}\equiv s_{24}$, $s_{14}\equiv s_{23}$ hold in
MARTY's convention. Imposing them shrinks the symbol set 6 → 3, and $s+t+u=\sum
m_i^2$ shrinks it to 2. I tested it: it does **not** merge any of the 30 / 11
templates, which means either the identification is false in this convention or no
two templates differ only by that swap. Do not assume it; test it.

### 2.3 The structured target: stop generating a string

The target is
$$\frac{N}{D},\qquad N=\sum_{k\le 5\,(8)} c_k\, \mu_k,\quad D = \big(\text{linear in } m^2,s + \tfrac12\,\text{reg\_prop}\big)^{-p}$$
with **at most 5 (QED) / 8 (QCD) monomials** and a denominator under 37 characters.

A typed head predicts this object directly:

- **denominator channel** — a small classification (which propagator, which power);
- **monomial support** — a multi-label prediction over the mass-dimension-4
  monomials in $\{m_i^2, s_{ij}\}$;
- **coefficients** — a rational-number head (numerator/denominator classification
  over the observed set, which is tiny, or a sign × log-magnitude regression).

Compare the two error surfaces. A sequence decoder must get ~110 tokens right in
order, and a single early slip is fatal — and it has no way to express "I know the
structure but not that coefficient". The structured head **cannot** emit a
malformed expression, decomposes the error into channel / support / coefficient
(diagnosis for free, `01` §6.1), and turns a 110-step autoregressive problem into a
handful of small predictions.

Predicted effect: this is where the accuracy lives, especially on QCD. It is also
the least "fashionable" change and the most likely to work.

*Risk to state:* it is a bet that the target family stays this rigid. It will not
survive to loop level, where $B_0, C_0, D_0$ Passarino-Veltman functions enter with
their own argument structure. The right posture is: structured head as the
primary model for tree level, sequence decoder maintained as the general fallback,
and the comparison between them is itself a reportable result.

---

## §3 Physics as constraint, not decoration

The proposal calls the work "physics-informed". Currently the physics is in the
prose. Here is how it becomes load-bearing, in increasing order of commitment.

### 3.1 Physics in the representation (cheap, high yield)
Canonicalisation, dummy-index normalisation, leg-role abstraction, prefix
serialisation. All measured above. Do these first.

### 3.2 Physics in the decoder (cheap, high yield)

**Mass-dimension homogeneity.** Measured: the expanded numerator is homogeneous of
mass dimension **exactly 4** in **360/360 QED and 234/234 QCD** records — 100%,
with weights $m_X\!\to\!1$, $s_{ij}\!\to\!2$, `reg_prop`$\to\!2$, coupling$\to\!0$.

This is an exact invariant, free to compute, and currently unused. It gives:

- a **decoding constraint** — at each step, mask tokens that cannot complete the
  current monomial to dimension 4. Whole error classes become unreachable.
- a **validity metric** — fraction of free-running outputs that are
  dimensionally consistent. Under constrained decoding this is 100% by
  construction; under free decoding it is one of the sharpest diagnostics
  available, because it separates "wrong physics" from "wrong arithmetic".
- a **loss term**, if constrained decoding is too rigid: penalise the expected
  dimension deviation under the model's own distribution.

Combined with a **typed prefix grammar** (arity known per operator, so the set of
legal next tokens is computable from the stack), the decoder can be made unable to
emit an unparsable string. Given a 30-symbol vocabulary, this is a small amount of
code for a large amount of guaranteed correctness.

### 3.3 Physics in the architecture (expensive, principled)

The group acting here is real: permutation of the two incoming and two outgoing
legs, together with the induced action on $\{s_{ij}\}$ and on the mass symbols;
plus crossing. A model equivariant to it would obtain the 360 → 30 collapse *by
construction* instead of learning it. Concretely: an encoder over the leg set with
permutation-equivariant attention and $s_{ij}$ as edge features, i.e. treat the
diagram as a graph over 4 external legs rather than as a flattened token string —
which is what the `graph_flat` serialisation currently fails to be (`02` §2.4:
the propagator is not even an edge).

This is the most principled change and the most work. It should be attempted only
after §2 and §3.2, and only if the diagnostics in §5 show the model failing at
exactly the symmetries it would enforce.

### 3.4 The physics-free control (mandatory)

Every claim above needs a matched control with the physics removed:
canonicalisation off; typed decoding off; leg-role abstraction off; a plain
character transformer at the same parameter count and budget. Without the control,
"physics-informed helps" is not a finding.

---

## §4 Architecture, from the parts you named

### 4.1 Embeddings

What must an embedding encode here? A token like `s_13` carries (a) an identity,
(b) a *type* (Mandelstam vs mass vs coupling vs digit vs operator), and (c) a
*structural role* (which argument of which operator). Types are known exactly from
the grammar — free supervision the current model discards by going to bytes.

Concretely: **factorise the embedding as identity ⊕ type ⊕ role**, with the type
channel taken from the parser rather than learned. Eight type rows. It costs
nothing and it makes §5.4 (does the MoE route by token class?) a measurable
question rather than a rhetorical one.

**On Role-Filler specifically.** `norm(C_e(A_e x + B_e r))` has no nonlinearity
between the linear maps, so it equals $(C_eA_e)x + (C_eB_e)r + b$ — a plain affine
map with $3d^2$ parameters doing the job of $2d^2$ (`02` §3.5). More importantly,
Smolensky's tensor-product representation binds role to filler by an **outer
product** $\sum_i r_i \otimes f_i$, whose defining property is that the binding is
*recoverable*: unbinding with a role vector retrieves its filler. An additive
composition is precisely the thing TPR was introduced to replace, because addition
destroys exactly that recoverability.

So either the name should change, or the module should. If the binding hypothesis
is what you want to test — and given §2.2 it is the most interesting hypothesis in
the project — implement a real binding: low-rank outer product, or Holographic
Reduced Representations (circular convolution, $O(d\log d)$), and then **test
unbinding**: given a hidden state, can a fixed unbinding operation recover the mass
symbol bound to leg 1? That is a sharp, falsifiable claim about the representation,
and it is exactly the "how humans learn" thread — variable binding is the classic
example of something humans do trivially and connectionist models do badly.

### 4.2 Mixture of Experts

Honest prior: **at this scale MoE will not move the number.** It addresses capacity
allocation, and capacity is not the binding constraint when 129.9M parameters face
324 examples and 30 target functions. Expect ≈0, possibly negative through routing
noise.

That does not make it worthless — it makes it a *hypothesis to test cheaply*
rather than a headline feature. To test it at all, three things are missing
(`02` §3.4): a load-balancing auxiliary loss, per-layer utilisation and routing-
entropy logging, and the mutual information $I(\text{token type};\text{expert})$.
With the type channel from §4.1 this is a direct measurement of Claim 3. Two
outcomes, both publishable: experts do partition by symbol class (the claim is
real, and you can show the partition), or they collapse (the claim is false at this
scale, and you say so — which is a more valuable contribution than a 1.2% gain on
n=36).

### 4.3 Attention: two different XSA operators

`Y ← Y − \frac{\langle Y, v_i\rangle}{\|v_i\|^2} v_i` removes, after softmax, the
component of position $i$'s output along position $i$'s own value vector. Masking
the diagonal of $QK^\top$ before softmax removes position $i$'s *weight* on itself
and renormalises the rest. These are different maps with different fixed points:
the first is a rank-1 projection applied to the output of a stochastic mixing; the
second changes the mixing distribution. The proposal's motivation describes the
second; the code implements the first.

Both are cheap. Build them as named arms (`vanilla`, `xsa_proj`, `xsa_mask`), and
test the claim against the operator the argument is about. Prediction: at n=36 the
difference will be inside the confidence interval either way — which is the
finding, and §4.4 offers a way to say something more precise than that.

### 4.4 NTK and the small-test-set problem

The deepest methodological problem in the current results is that architectural
claims are being decided on 36 and 24 test points. "94.4% vs 88.9%" on n=36 is a
difference of two records. No amount of careful training fixes that.

There is a way out that does not require more data: **measure the inductive bias
directly instead of inferring it from test accuracy.**

In the overparameterised regime the network is well approximated near
initialisation by its linearisation
$$f(x;\theta)\approx f(x;\theta_0)+\nabla_\theta f(x;\theta_0)^\top(\theta-\theta_0),\qquad
K(x,x')=\big\langle \nabla_\theta f(x;\theta_0),\ \nabla_\theta f(x';\theta_0)\big\rangle$$
so the architecture's bias *is* the kernel $K$. Gradient-descent generalisation
under this kernel is governed by how well the kernel's top eigendirections align
with the target — the **kernel–target alignment**
$$A(K,Y)=\frac{\langle K,\;YY^\top\rangle_F}{\|K\|_F\,\|YY^\top\|_F}$$
with $Y$ the one-hot template-class matrix from §1.3 (30 or 11 classes).

The point: $K$ is computed on the **324 training points**, not on 36 test points.
Comparing $A(K_{\text{vanilla}}, Y)$, $A(K_{\text{xsa}}, Y)$,
$A(K_{\text{role-filler}}, Y)$ gives a measurement of "does this component encode
the right bias?" with an order of magnitude more statistical power than the test
set, and with no leakage at all. If XSA raises alignment with the template
structure, that is a *mechanistic* argument for it, independent of whether two
extra records flipped.

**Caveats, stated plainly, because this is easy to over-claim:**

- NTK constancy is a large-width, small-learning-rate statement. At $d=512$ with
  4-layer stacks the kernel drifts during training. Measure both $K$ at
  initialisation and the empirical after-training kernel, and report the drift —
  the drift itself is informative.
- MoE routing is **discrete**, so $f$ is piecewise smooth and the tangent kernel is
  only locally constant, jumping at routing boundaries. NTK statements about the
  MoE arms are weaker than about the dense arms. Say so.
- $f$ is sequence-valued, so the naive kernel is $(nT)\times(nT)$. Use a scalar
  readout — the summed log-likelihood of the correct sequence — to get a tractable
  $n\times n$ kernel, and state that the choice of readout matters.
- A much cheaper proxy, worth running first: the **CKA between the encoder
  representation Gram matrix and the class kernel $YY^\top$**. Same question, no
  gradients, minutes instead of hours. If CKA already separates the arms, the full
  NTK is confirmation rather than discovery.

Second, unrelated theoretical point worth carrying: with $\varepsilon=0.1$ label
smoothing over $V=260$, the loss floor is
$$H(q)=-\big[q_1\log q_1+(V-1)q_0\log q_0\big]=0.8778,\quad q_1=1-\varepsilon+\tfrac{\varepsilon}{V}$$
and the QED run reached 0.8828. Near this floor the per-token gradient is
$O(10^{-3})$: the *optimisation* had effectively stopped while exact match still
climbed 75% → 94%. That decoupling is not a curiosity, it is the reason model
selection failed (`02` §5.1), and it is a clean argument for selecting on a
task metric rather than a smoothed likelihood on a deterministic problem.

### 4.5 Capacity

129.9M parameters, 324 examples, 30 target functions. Run the sweep
($d_{\text{model}} \in \{128,256,512\}$, layers $\in \{2,4\}$) and expect the small
model to win. Add dropout where it was declared but never applied (`02` §3.8).
This is boring and it will probably beat every architectural idea in this document
on the headline number.

---

## §5 Evaluation and training-time diagnostics

This is the part of the objective the proposal names ("diagnose and evaluate the
models more professionally and more incrementally") and the part the code has
least of. Currently the only training signals are three scalars.

### 5.1 Decompose accuracy instead of reporting it

Given the structure in §1.3, exact match factorises approximately as
$$\text{EM}\;\approx\;P(\text{correct template})\times P(\text{correct binding}\mid\text{template})\times P(\text{correct coefficients}\mid\cdot)$$
Every factor is measurable by decoding, canonicalising, and mapping to leg-role
form. A single 66.7% becomes "picks the right template 91% of the time, binds
masses correctly 88% given that, gets coefficients right 83% given both" — which
tells you what to fix. This is the highest-value diagnostic in the document and it
requires no model change at all.

### 5.2 Per-position, per-type, per-class loss

Three curves the current code cannot produce:
- **loss vs position** — is the model failing early (structure) or late (tail
  coefficients)? Under a 63-byte truncation this was unanswerable by construction.
- **loss vs token type** — using the type channel from §4.1. Are operators easy and
  coefficients hard, or the reverse?
- **accuracy per template class** — with 30 classes and sizes ranging 36 down to 3,
  a per-class breakdown immediately shows whether the model is just failing the
  rare classes. Class sizes are `36,36,36,18×9,12×3,6×3,3×12`; a random 10% test
  set will contain the size-3 classes erratically or not at all, which is a large
  and currently invisible source of run-to-run variance.

### 5.3 Probing: does the graph pathway encode what we claim?

Train a linear probe on frozen encoder states at each layer to predict:
propagator channel ($s/t/u$), the leg masses, the coupling order, the template
class. If a linear probe recovers the channel from the graph encoder at layer 2,
the graph pathway demonstrably carries topological information — which is Claim 1,
tested properly, on 324 points instead of 36. If it cannot, Claim 1 is false and
that is worth knowing.

### 5.4 Representation geometry: test the interference claim directly

The stated purpose of the dual pathway is to "prevent early modality interference".
That is a claim about representations and should be measured as one: **CKA between
the graph-encoder and math-encoder representations, layer by layer**, for the dual
model versus a single-stream control. If the streams stay decorrelated in the dual
model and collapse in the single-stream one, the claim is supported directly,
without routing through a noisy test metric.

Similarly for XSA: compare the attention entropy and the effective rank of the
attention matrices between `vanilla`, `xsa_proj`, and `xsa_mask`. The claimed
mechanism is "stop tokens copying themselves"; that has a signature in the
attention spectrum, and it is measurable on the training set.

### 5.5 Replace the heatmaps with a statistic

Four eyeballed heatmaps computed on the *training* set (`02` §5.4) produced "no
consistent interpretable patterns", which was inevitable. The quantitative version:
for each output token, the **fraction of cross-attention mass falling on the graph
half of the memory**, aggregated over the whole validation set and stratified by
output token type. That answers the actual question — *when* does the decoder
consult the diagram rather than the algebra? — as a curve with error bars instead
of a picture. If a gated fusion is added (`01` §4.6), the gate value $g$ is an even
more direct readout of the same thing.

### 5.6 Memorisation vs composition, measured

- **Leave-one-template-out** across all 30 / 11 classes. The mean is the honest
  generalisation number and $n$ is the class count, not the record count.
- **Retrieval-margin**: for each test item, the model's accuracy versus the
  1-NN retrieval baseline's accuracy on the same item. Items where retrieval
  succeeds and the model fails are diagnostic; items where the model succeeds and
  retrieval fails are the actual evidence of composition.
- **Training-set influence**: for a correct test prediction, which training records
  most reduce its loss? If it is always the same-template records, the model is
  interpolating; if it is records sharing sub-structure across templates, it is
  composing.

### 5.7 The reporting standard

Every number: metric name, decoding mode, split protocol, $n$, 95% Wilson interval,
and the seed count. Three seeds minimum for any comparison. No maxima over epochs.
No teacher-forced number ever called "exact match".

---

## §6 Interpretable vs non-interpretable: run both, deliberately

Three model families, all supported by the blueprint in `01`, differing in where
the interpretability lives:

**A — Sequence decoder (the current design, fixed).** Canonical prefix target, no
truncation, typed constrained decoding. Interpretable only *post hoc*: probes,
attention statistics, CKA. General — extends to loop level and EW without redesign.
This is the workhorse and the SYMBA-comparable baseline.

**B — Retrieve-and-substitute.** An explicit template codebook (learned, or
initialised from the 30 / 11 discovered classes) with a retrieval head over it, plus
a binding head that maps leg roles to mass symbols and predicts the charge/colour
prefactor. **Interpretable by construction**: for every prediction you can read off
which template was selected, with what confidence, and what was bound into it.

The objection to B is obvious and should be met head-on rather than hidden: it is
memorisation made architectural. That is precisely why it is worth building. It
provides (i) a near-tight upper bound on what pure retrieval achieves on this
corpus, and therefore (ii) the exact quantity that model A must exceed for
"composition" to be a defensible word. A project whose corpus is 30 functions
either measures the retrieval ceiling or is guessing about it.

**C — Structured / typed head** (§2.3). Predicts denominator channel, monomial
support, and coefficients. Interpretable in its factors, guaranteed well-formed,
and its error decomposes without any extra analysis. Likely the accuracy winner at
tree level; likely the first to break at loop level.

The comparison A vs B vs C — accuracy, generalisation under protocol B,
interpretability, and extensibility — is a stronger contribution than any single
number, and it is exactly what the SYMBA project would find useful.

---

## §7 What I predict will actually move the number

Ranked by expected effect on held-out accuracy, with the reasoning and the risk.

| # | change | expected effect | why | risk |
|---|---|---|---|---|
| 1 | remove truncation + canonical target | **redefines the task** | 100% of targets are currently truncated | EM will *drop* initially — the task got harder. Do not read that as regression |
| 2 | typed + dimension-constrained decoding | **large** | 30-symbol vocabulary, exact dim-4 invariant; whole error classes become unreachable | none material |
| 3 | structured head (§2.3) | **large**, esp. QCD | ≤8 monomials vs ~110 autoregressive steps | brittle to loop level |
| 4 | capacity ↓ + real dropout | **moderate–large** | 129.9M params / 324 examples | none |
| 5 | select on symbolic EM, not smoothed CE | **moderate**, free | loss floor 0.8778, reached 0.8828 | none |
| 6 | wire in AST + graph (the actual fix) | **unknown — this is the experiment** | never tested | may show the structure buys nothing, which is a real result |
| 7 | exact flavour augmentation (§2.2) | **moderate–large** | prefactor $=\tfrac14(Q_aQ_b)^2$ verified 216/216 | must verify the photon-leg rule before using those 144 |
| 8 | beam length normalisation + KV cache | **small accuracy, large speed** | short-sequence bias; 30–100× decode speedup | none |
| 9 | padding masks | **small–moderate** | attention currently runs over PAD | none |
| 10 | fix the propagator self-loop | **small–moderate** | the graph has no internal edge at all | none |
| 11 | XSA variants | **≈0 at this n** | difference will sit inside the CI | worth running as an NTK/CKA measurement, not for the number |
| 12 | MoE (balanced, instrumented) | **≈0, possibly negative** | capacity is not the constraint here | valuable as a measurement of Claim 3 |

Items 1–5 are the accuracy programme. Item 6 is the scientific programme. Items
11–12 are the paper's honest negative results, and reporting them well is worth
more than a fractional gain on 36 records.

---

## §8 What could make this analysis wrong

- **The 30/11 class count depends on my abstraction** (mass symbol → leg index of
  first occurrence). A different, equally defensible abstraction could give a
  different count. The collapse is large enough (12× and 21×) that the conclusion
  is robust, but the exact number should be re-derived independently.
- **Canonicalisation may not be the right target** if the downstream consumer
  needs MARTY's specific form. That is a question for your mentors, and it is a
  question, not an assumption — which is why `01` §6.1 reports both metrics.
- **The prefactor rule was verified on QED four-fermion processes only** (216/216,
  exactly). The 144 photon-leg records follow a different rule that I have not
  derived, and QCD adds colour factors. Do not augment until each rule is verified
  on its own subset.
- **The Mandelstam identification is unverified** and did not collapse any
  templates. Treat §2.2's last paragraph as an open question, not a result.
- **Everything here is tree-level, 2→2.** The structured head (§2.3) and the
  dimension constraint (§3.2) both lean on the target being a low-order rational
  function. Loop level introduces $B_0,C_0,D_0$ and the argument weakens. The
  sequence model (family A) is the insurance policy and should not be abandoned
  even if C wins at tree level.
