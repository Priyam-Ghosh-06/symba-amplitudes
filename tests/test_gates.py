"""The gate table of 01 SS7. Nothing downstream is trusted until these pass.

    python tests/test_gates.py          # no pytest needed
    pytest tests/ -q                    # also works

G9 and G12 are the two that matter most: the first asserts the model is fed the
parsed streams and not the raw strings, the second that train/val/test are
disjoint. Both are a few lines and both invalidate or validate every number the
project will ever produce.
"""

import os
import sys

import sympy
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symba.config import Config, PAD, SOS, EOS
from symba.data.ast_parse import amp_to_prefix
from symba.data.canonical import (canonicalise, check_mass_dimension,
                                  monomial_dimensions, to_sympy,
                                  verify_equivalence)
from symba.data.dataset import AmplitudeDataset, collate
from symba.data.graph import FeynmanGraph, parse_legs
from symba.data.load import load_theory
from symba.data.normalize import standardize, strip_keywords
from symba.data.pipeline import _budget, build, seed_run
from symba.data.serialize import (PrefixState, from_prefix, is_well_formed,
                                  target_tokens, to_prefix, token_type)
from symba.data.splits import assert_disjoint, split_records
from symba.data.vocab import Vocab
from symba.experiment import ARMS
from symba.eval.decode import ConstraintMask, beam_search
from symba.eval.metrics import (COMPLEXITY_CAP, TERM_CAP,
                                evaluate_predictions, expansion_terms,
                                wilson)
from symba.model.model import AmplitudeModel

DATA_ROOT = "data/Symba"
_CACHE = {}


def bundle(theory="QED"):
    """Built once and reused; the pipeline is deterministic given the seed."""
    if theory not in _CACHE:
        cfg = Config().with_overrides(**{"data.theory": theory})
        _CACHE[theory] = (cfg, build(cfg, verbose=False))
    return _CACHE[theory]


# --- G1: parse and validate --------------------------------------------------

def test_G1_one_record_per_nonblank_line():
    for theory, expected in (("QED", 360), ("QCD", 234)):
        records = load_theory(DATA_ROOT, theory)
        assert len(records) == expected, f"{theory}: {len(records)}"
        # Every record has exactly 4 external legs.
        for record in records[:50]:
            in_legs, out_legs = parse_legs(strip_keywords(record.interaction))
            assert len(in_legs) + len(out_legs) == 4


# --- G2: normalisation is idempotent and index-safe --------------------------

def test_G2_standardize_is_idempotent():
    raw = r"%\sigma_18923 * %\mu_77 + %\sigma_18923"
    once = standardize(raw)
    assert once == standardize(once)
    assert "18923" not in once and "_d1" in once
    # Physical symbols carry no % prefix and must survive untouched.
    assert standardize("s_12 * m_e") == "s_12 * m_e"


def test_G2_no_digit_suffixed_dummy_survives():
    _cfg, b = bundle()
    import re
    pattern = re.compile(r"%[a-zA-Z\\]+_\d+")
    for record in b.records[:100]:
        assert not pattern.search(record["amp_std"])


# --- G3 / G4: canonicalisation is faithful and dimensionally homogeneous -----

def test_G3_canonical_equals_raw():
    _cfg, b = bundle()
    for record in b.records[:40]:
        assert verify_equivalence(record["canon_num"], record["canon_den"],
                                  record["sq_amp_std"]), record.line_no


def test_G4_numerator_is_mass_dimension_four():
    for theory in ("QED", "QCD"):
        _cfg, b = bundle(theory)
        for record in b.records:
            assert record["mass_dim_ok"], f"{theory}:{record.line_no}"
    # And the check actually discriminates: a dimension-3 numerator fails.
    assert not check_mass_dimension(to_sympy("m_e^3"))
    assert check_mass_dimension(to_sympy("s_12*s_34"))


# --- G5 / G13: serialisation round-trips -------------------------------------

def test_G5_prefix_roundtrip_is_symbolically_equal():
    for theory in ("QED", "QCD"):
        _cfg, b = bundle(theory)
        for record in b.records[:60]:
            tokens = record["target_tokens"]
            assert sympy.simplify(
                from_prefix(tokens) - record["canon_expr"]) == 0, record.line_no


def test_G5_prefix_has_no_brackets_and_splits_digits():
    tokens = to_prefix(to_sympy("16*m_e^2 + 1"))
    assert "(" not in tokens and ")" not in tokens
    assert "INT+" in tokens and "1" in tokens and "6" in tokens


def test_G13_decode_encode_is_identity_on_ground_truth():
    _cfg, b = bundle()
    vocab = b.target_vocab
    for record in b.train[:60]:
        ids = vocab.encode(record["target_tokens"])
        assert vocab.decode(ids[1:]) == record["target_tokens"]


def test_malformed_prefix_is_rejected():
    assert is_well_formed(["+", "m_e", "s_12"])
    assert not is_well_formed(["+", "m_e"])            # missing an operand
    assert not is_well_formed(["m_e", "s_12"])         # trailing token
    assert not is_well_formed(["INT+"])                # marker with no digits


# --- G6: the graph is a real, connected edge list ----------------------------

def test_G6_propagators_join_distinct_vertices():
    for theory in ("QED", "QCD"):
        _cfg, b = bundle(theory)
        for record in b.records:
            graph = record["graph"]
            assert graph.edges, f"{theory}:{record.line_no} has no propagator"
            for a, vertex_b, _p in graph.edges:
                assert a != vertex_b, f"self-loop at {a}"
            assert graph.is_connected()


def test_G6_vertex_order_is_canonical():
    """Two spellings of the same diagram must serialise identically."""
    interaction = "e(X) AntiPart e(X) to mu(X) AntiPart mu(X)"
    a = FeynmanGraph(interaction,
                     "V_0:e(X_1), AntiPart e(X_2), OffShell A(V_0), "
                     "V_1:mu(X_3), AntiPart mu(X_4), OffShell A(V_1),")
    b = FeynmanGraph(interaction,
                     "V_1:mu(X_3), AntiPart mu(X_4), OffShell A(V_1), "
                     "V_0:e(X_1), AntiPart e(X_2), OffShell A(V_0),")
    assert a.to_tokens() == b.to_tokens()


# --- G7 / G8: vocabulary comes from the training split only ------------------

def test_G8_vocab_ids_are_deterministic_and_specials_fixed():
    _cfg, b = bundle()
    vocab = b.target_vocab
    assert vocab.itos[:4] == ["<pad>", "<sos>", "<eos>", "<unk>"]
    assert vocab.itos[4:14] == [str(d) for d in range(10)]
    # Rebuilding from the same records gives the identical mapping.
    again = Vocab(r["target_tokens"] for r in b.train)
    assert again.itos == vocab.itos


def test_G7_no_training_token_reaches_the_encoder_as_unk():
    """The vocabulary must cover the stream the model actually reads.

    ``amp_segments`` is what the math encoder is fed, and it carries a
    ``<diagrams>`` placeholder that never appears in ``amp_tokens``. Building
    the vocabulary over ``amp_tokens`` sent that token to the encoder as
    ``<unk>`` in all 234 QCD records, and the OOV gate could not see it
    because it measured the wrong stream.
    """
    for theory in ("QED", "QCD"):
        _cfg, b = bundle(theory)
        for record in b.train:
            for segment in record["amp_segments"]:
                unknown = [t for t in segment if t not in b.amp_vocab.stoi]
                assert not unknown, f"{theory}: {unknown[:3]} encode as <unk>"
            for token in record["graph_tokens"]:
                assert token in b.graph_vocab.stoi, f"{theory}: {token!r}"


def test_G8_vocab_never_sees_val_or_test():
    _cfg, b = bundle()
    train_tokens = {t for r in b.train for t in r["target_tokens"]}
    extras = set(b.target_vocab.itos[14:]) - train_tokens
    assert not extras, f"vocabulary contains non-train tokens: {extras}"


def test_G7_oov_is_measured_not_hidden():
    _cfg, b = bundle()
    for key, entry in b.stats["oov"].items():
        assert 0.0 <= entry["rate"] < 0.05, (key, entry)


# --- G9: the model is fed the parsed streams, not the raw strings ------------

def test_G9_tensors_derive_from_parsed_fields():
    """The test the previous code fails.

    Perturbing the raw ``amp`` string must not change the model input;
    perturbing ``amp_tokens`` must.
    """
    _cfg, b = bundle()
    records = b.train[:4]

    def encode(recs):
        ds = AmplitudeDataset(recs, b.graph_vocab, b.amp_vocab, b.target_vocab)
        return collate([ds[i] for i in range(len(ds))])["amp"]

    before = encode(records)

    records[0].amp = records[0].amp + " + 999*m_e"      # raw field only
    records[0]["amp_std"] = records[0]["amp_std"] + " + 999*m_e"
    assert torch.equal(encode(records), before), \
        "model input changed when only the RAW amp was perturbed"

    original = list(records[0]["amp_tokens"])
    records[0]["amp_tokens"] = original + ["m_e"]
    assert not torch.equal(encode(records), before), \
        "model input did NOT change when the PARSED amp was perturbed"
    records[0]["amp_tokens"] = original


def test_G9_decoding_cannot_see_the_target():
    """The decisive no-leak gate.

    G9 above proves the *tensors* come from the parsed streams. This proves the
    stronger and more important thing: at prediction time the ground-truth
    target is not an input. Corrupting, blanking, or deleting ``batch["target"]``
    must leave the decoded sequences bit-identical, because free-running decode
    starts from SOS and conditions only on the encoder memory and its own
    output. If any of these change the prediction, every accuracy number in the
    project is meaningless.
    """
    cfg, b = bundle()
    torch.manual_seed(0)
    model = AmplitudeModel(cfg.model, b.graph_vocab, b.amp_vocab,
                           b.target_vocab, b.lengths,
                           segment_amp=b.segment_amp,
                           segment_len=b.segment_len).eval()
    batch = next(iter(b.loaders["test"]))
    constraint = ConstraintMask(b.target_vocab)
    max_len = b.decode_budget

    def decode(payload):
        return beam_search(model, payload, b.target_vocab, beam_width=4,
                           max_len=max_len, length_penalty=0.7,
                           constrained=True, constraint=constraint)

    baseline = decode(batch)
    n_target = len(b.target_vocab)

    blanked = dict(batch, target=torch.full_like(batch["target"], PAD))
    randomised = dict(batch, target=torch.randint(4, n_target,
                                                  batch["target"].shape))
    dropped = {k: v for k, v in batch.items() if k != "target"}

    for name, payload in (("blanked", blanked), ("randomised", randomised),
                          ("absent", dropped)):
        assert decode(payload) == baseline,             f"decoding changed when the target was {name} - the target leaks"

    # The encoder must not read it either.
    memory, _mask, _n = model.encode(batch)
    other, _mask2, _n2 = model.encode(randomised)
    assert torch.equal(memory, other), "encode() depends on the target"


def test_G9_decoder_self_attention_is_causal():
    """Teacher forcing must not let position t see position t+1.

    Randomising the tail of the target has to leave the logits for the earlier
    positions untouched, and has to change the later ones.
    """
    cfg, b = bundle()
    torch.manual_seed(0)
    model = AmplitudeModel(cfg.model, b.graph_vocab, b.amp_vocab,
                           b.target_vocab, b.lengths,
                           segment_amp=b.segment_amp,
                           segment_len=b.segment_len).eval()
    batch = next(iter(b.loaders["train"]))
    cut = 5

    with torch.no_grad():
        before = model(batch)["logits"]
        tail = batch["target"].clone()
        tail[:, cut:] = torch.randint(4, len(b.target_vocab),
                                      tail[:, cut:].shape)
        after = model(dict(batch, target=tail))["logits"]

    assert torch.allclose(before[:, :cut - 1], after[:, :cut - 1], atol=1e-5),         "logits before the cut moved - the decoder attends to the future"
    assert not torch.allclose(before[:, cut:], after[:, cut:], atol=1e-5),         "logits after the cut did not move - the test is not exercising anything"


def test_G10_decode_budget_comes_from_the_training_split_only():
    """What the decoder may emit must not be set by held-out data.

    The decode budget caps the output, so a maximum taken over all three splits
    would let held-out target lengths reach the thing under test. Positional
    table sizes are a separate quantity and are deliberately *not* asserted to
    be train-only: an embedding row no gradient touches carries no information
    about a held-out target, and sizing tables from train alone would make a
    longer held-out sequence index off the end.
    """
    _cfg, b = bundle()
    train_max = b.datasets["train"].lengths()
    assert b.decode_budget == _budget(train_max[2]),         "decode budget is not derived from the training split alone"
    for i, name in enumerate(("graph", "amp", "target")):
        assert b.lengths[i] >= train_max[i], f"{name} table below train max"


def test_G10_overlong_heldout_target_is_a_miss_not_a_crash():
    """A held-out target the decoder cannot emit is a finding, not an error.

    Raising would kill a whole leave-one-template-out fold whenever a held-out
    class happens to be longer than anything in training. The model
    structurally cannot emit such a target, so it scores as a miss on its own;
    the build only has to count it.
    """
    _cfg, b = bundle()
    assert "unreachable_targets" in b.stats
    for split in ("val", "test"):
        counted = b.stats["unreachable_targets"][split]
        actual = sum(1 for item in b.datasets[split].items
                     if item["target"].size(0) > b.decode_budget)
        assert counted == actual, f"{split}: {counted} counted, {actual} present"


def test_G14_equal_implies_structure():
    """symbolic EM = structure x coefficient has to be an identity.

    It only holds if ``equal`` implies ``structure``. It did not: the reference
    is normalised expand -> together -> cancel while a prediction was only put
    through ``together``, so an equal-but-uncancelled prediction kept a
    denominator the reference lacks and scored equal-but-structure-false,
    breaking the identity downward. Both sides are now cancelled, and the
    channel test is projective.
    """
    # Algebraically identical to 4*s_12/s_13, written four awkward ways.
    reference = ["/", "*", "INT+", "4", "s_12", "s_13"]
    rewrites = [
        ["/", "*", "*", "INT+", "2", "INT+", "2", "s_12", "s_13"],     # 2*2
        ["/", "*", "INT+", "8", "s_12", "*", "INT+", "2", "s_13"],     # 8N/2D
        ["/", "*", "INT+", "4", "s_12", "s_13"],                       # as-is
    ]
    scored = evaluate_predictions(rewrites, [reference] * len(rewrites))
    equal = scored["per_record"]["symbolic"]
    structure = scored["per_record"]["structure"]
    broken = [i for i, (e, st) in enumerate(zip(equal, structure)) if e and not st]
    assert not broken, f"equal but not structure at {broken} - identity broken"
    assert all(equal), "these are all the same rational function"

    # And the identity itself, on the aggregate.
    em = scored["symbolic_exact_match"]["value"]
    product = (scored["structure_exact"]["value"]
               * scored["coefficient_exact"]["value"])
    assert abs(em - product) < 1e-9, f"{em} != {product}"


def test_G14_expansion_bound_catches_what_count_ops_misses():
    """The complexity guard must bound expansion cost, not expression size.

    ``count_ops`` of an unexpanded product of k binomials is constant in k
    while the expansion grows as 2**k, and OPERATOR_SLACK admits roughly
    k = 60. The guard has to look at the quantity that actually blows up.
    """
    a, b_, c, d = (sympy.Symbol(n) for n in ("s_12", "s_13", "s_14", "s_23"))
    wide = ((a + b_) * (c + d) * (a + c) * (b_ + d)
            * (a + d) * (b_ + c) * (a + b_ + c) * (b_ + c + d))

    assert sympy.count_ops(wide) < COMPLEXITY_CAP,         "count_ops was expected to under-report this"
    assert expansion_terms(wide) > sympy.count_ops(wide),         "the expansion bound must exceed the written size here"
    assert expansion_terms(sympy.sympify("4*s_12/s_13")) <= 2,         "a legitimate canonical target must stay cheap"


def test_G14_per_record_vector_matches_the_aggregate():
    """The stored per-record vector must be the thing the headline sums."""
    _cfg, b = bundle()
    references = [r["target_tokens"] for r in b.test[:12]]
    predictions = list(references)
    predictions[0] = ["s_12"]                      # force one miss
    scored = evaluate_predictions(predictions, references)

    flags = scored["per_record"]["symbolic"]
    assert len(flags) == len(references)
    assert sum(flags) == scored["symbolic_exact_match"]["successes"]


def test_G14_expansion_bound_accounts_for_together():
    """The guard must bound what _as_fraction computes, not expand() alone.

    _as_fraction runs together() before expand(), and together() multiplies
    every other denominator into each term of a sum of fractions. Counting
    1/(a + b) as one term let a sum of k such fractions through at any k:
    measured, 0.9 s to score at k = 8 and 7.5 s at k = 10, tripling per
    fraction. With the bound fixed, k = 12 has to be rejected and cheap; if it
    regresses this test takes over a minute and then fails.
    """
    import time
    _cfg, b = bundle()
    reference = max((r["target_tokens"] for r in b.test),
                    key=lambda t: sum(tok in ("+", "*", "/", "^", "NEG")
                                      for tok in t))
    syms = ["s_12", "s_13", "s_14", "s_23", "s_24", "s_34", "m_e", "m_mu"]
    pairs = [(x, y) for i, x in enumerate(syms) for y in syms[i + 1:]]
    k = 12
    prediction = ["+"] * (k - 1)
    for x, y in pairs[:k]:
        prediction += ["/", "s_12", "+", x, y]

    assert expansion_terms(from_prefix(prediction)) > TERM_CAP
    start = time.time()
    scored = evaluate_predictions([prediction], [reference])
    assert time.time() - start < 5, "sum of fractions was expanded, not rejected"
    assert scored["complexity_bailouts"] == 1


def test_G14_coefficient_metric_is_not_symbolic_em_again():
    """The structure/coefficient split has to separate two real failure modes.

    ``coefficient_exact`` used to be ``equal and same_monomials``, which is
    implied by ``equal``, so it reproduced symbolic EM in all 46 archived runs.
    A prediction with the right structure and a wrong number must now count as
    a structure hit and a coefficient miss.
    """
    right = ["/", "*", "INT+", "4", "s_12", "s_13"]          # 4*s_12/s_13
    wrong_number = ["/", "*", "INT+", "5", "s_12", "s_13"]   # 5*s_12/s_13
    wrong_form = ["/", "*", "INT+", "4", "m_e", "s_13"]      # 4*m_e/s_13

    scored = evaluate_predictions([wrong_number], [right])
    assert scored["symbolic_exact_match"]["value"] == 0.0
    assert scored["structure_exact"]["value"] == 1.0,         "right monomials and right channel should count as correct structure"
    assert scored["coefficient_exact"]["value"] == 0.0

    scored = evaluate_predictions([wrong_form], [right])
    assert scored["structure_exact"]["value"] == 0.0
    # Conditional metric: no structurally correct prediction, so n is 0.
    assert scored["coefficient_exact"]["n"] == 0


def test_G9_target_is_the_canonical_form():
    _cfg, b = bundle()
    record = b.train[0]
    assert sympy.simplify(
        from_prefix(record["target_tokens"]) - record["canon_expr"]) == 0


# --- G10: nothing is truncated, masks match true lengths ---------------------

def test_G10_no_truncation_and_masks_are_exact():
    _cfg, b = bundle()
    batch = next(iter(b.loaders["train"]))
    for key, mask_key in (("graph", "graph_mask"), ("amp", "amp_mask"),
                          ("target", "target_mask")):
        ids, mask = batch[key], batch[mask_key]
        assert torch.equal(mask, ids.eq(PAD))
        for row in range(ids.size(0)):
            true_len = int((~mask[row]).sum())
            assert ids[row, 0] == SOS
            assert ids[row, true_len - 1] == EOS, "sequence lost its EOS"

    # Over-budget input raises rather than clipping.
    try:
        AmplitudeDataset(b.train[:2], b.graph_vocab, b.amp_vocab,
                         b.target_vocab, max_lengths=(4, 4, 4))
    except ValueError:
        pass
    else:
        raise AssertionError("over-budget sequence was silently accepted")


# --- G11: ablation arms are effective ----------------------------------------

def test_G11_modality_ablation_changes_the_model():
    cfg, b = bundle()
    full = AmplitudeModel(cfg.model, b.graph_vocab, b.amp_vocab,
                          b.target_vocab, b.lengths)
    math_only = AmplitudeModel(
        cfg.model.__class__(**{**vars(cfg.model), "use_graph": False}),
        b.graph_vocab, b.amp_vocab, b.target_vocab, b.lengths)

    assert full.graph_encoder is not None
    assert math_only.graph_encoder is None, "ablated pathway still built"
    assert math_only.n_parameters() < full.n_parameters()

    batch = next(iter(b.loaders["train"]))
    memory_full, _m, graph_len = full.encode(batch)
    memory_math, _m2, graph_len_math = math_only.encode(batch)
    assert graph_len > 0 and graph_len_math == 0
    assert memory_math.size(1) < memory_full.size(1)


def test_G11_attention_arms_are_different_operators():
    cfg, b = bundle()
    batch = next(iter(b.loaders["train"]))
    outputs = {}
    for kind in ("vanilla", "xsa_proj", "xsa_mask"):
        torch.manual_seed(0)
        model = AmplitudeModel(
            cfg.model.__class__(**{**vars(cfg.model), "attention": kind}),
            b.graph_vocab, b.amp_vocab, b.target_vocab, b.lengths)
        model.eval()
        outputs[kind] = model(batch)["logits"]

    assert not torch.allclose(outputs["vanilla"], outputs["xsa_proj"])
    assert not torch.allclose(outputs["vanilla"], outputs["xsa_mask"])
    assert not torch.allclose(outputs["xsa_proj"], outputs["xsa_mask"])


def test_G11_tied_embeddings_survive_init():
    cfg, b = bundle()
    tied = AmplitudeModel(cfg.model, b.graph_vocab, b.amp_vocab,
                          b.target_vocab, b.lengths)
    assert (tied.fc_out.weight.data_ptr()
            == tied.decoder_embed.token_embed.weight.data_ptr())


def test_G11_padding_is_ignored_by_the_encoder():
    """Extra padding must not change the representation of real positions."""
    cfg, b = bundle()
    torch.manual_seed(0)
    model = AmplitudeModel(cfg.model, b.graph_vocab, b.amp_vocab,
                           b.target_vocab, b.lengths).eval()

    batch = next(iter(b.loaders["train"]))
    short = model.graph_encoder(batch["graph"], batch["graph_mask"])

    pad = torch.full((batch["graph"].size(0), 5), PAD, dtype=torch.long)
    padded_ids = torch.cat([batch["graph"], pad], dim=1)
    padded = model.graph_encoder(padded_ids, padded_ids.eq(PAD))

    true_len = int((~batch["graph_mask"][0]).sum())
    assert torch.allclose(short[0, :true_len], padded[0, :true_len], atol=1e-5)


# --- G12: the three splits are disjoint --------------------------------------

def test_G12_a_run_is_a_function_of_its_config_and_seed():
    """The reproducibility gate. Without it every regenerated table is junk.

    ``build`` seeds once, but a job trains several arms against a cached
    bundle, so the global RNG has been advanced by every model, dropout draw
    and shuffle that came before. The same (arm, seed) therefore produced
    different initial weights depending on its position in the job - measured
    at 0.53 max absolute difference in the decoder embedding, which is not a
    rounding artefact.

    ``seed_run`` resets immediately before construction. This builds the same
    arm twice with an unrelated model in between and demands bit-identical
    weights.
    """
    cfg, b = bundle()
    run_cfg = cfg.with_overrides(**ARMS["full_vanilla_dense"])

    def build_model(c):
        seed_run(c, b)
        return AmplitudeModel(c.model, b.graph_vocab, b.amp_vocab,
                              b.target_vocab, b.lengths,
                              segment_amp=b.segment_amp,
                              segment_len=b.segment_len)

    first = build_model(run_cfg)
    build_model(cfg.with_overrides(**ARMS["capacity_64"]))   # perturb the RNG
    second = build_model(run_cfg)

    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key]),             f"{key} differs between two builds of the same (arm, seed)"

    # The shuffle stream must reset too, or arm 2 sees a different epoch order.
    seed_run(run_cfg, b)
    first_order = list(iter(b.loaders["train"].batch_sampler))
    seed_run(run_cfg, b)
    assert list(iter(b.loaders["train"].batch_sampler)) == first_order


def test_G12_splits_are_disjoint_and_stable():
    _cfg, b = bundle()
    assert assert_disjoint(b.train, b.val, b.test)
    assert len(b.train) + len(b.val) + len(b.test) == len(b.records)

    # Hash-based assignment: reordering the records cannot change the split.
    shuffled = list(reversed(b.records))
    a_train, a_val, a_test = split_records(b.records, "record", 0.1, 0.1, 0)
    c_train, c_val, c_test = split_records(shuffled, "record", 0.1, 0.1, 0)
    for x, y in ((a_train, c_train), (a_val, c_val), (a_test, c_test)):
        assert {id(r) for r in x} == {id(r) for r in y}


def test_G12_template_split_keeps_classes_whole():
    _cfg, b = bundle()
    train, val, test = split_records(b.records, "template", 0.15, 0.15, 0)
    parts = [{r["template"] for r in part} for part in (train, val, test)]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2])


# --- G14: the metrics themselves are correct ---------------------------------

def test_G14_metrics_self_test():
    _cfg, b = bundle()
    references = [r["target_tokens"] for r in b.test[:20]]
    templates = [r["template"] for r in b.test[:20]]

    perfect = evaluate_predictions(list(references), references, templates)
    assert perfect["symbolic_exact_match"]["value"] == 1.0
    assert perfect["sequence_exact_match"]["value"] == 1.0
    assert perfect["parse_validity"]["value"] == 1.0
    assert perfect["mass_dimension_validity"]["value"] == 1.0

    garbage = [["m_e"] for _ in references]
    scored = evaluate_predictions(garbage, references, templates)
    assert scored["symbolic_exact_match"]["value"] == 0.0
    assert scored["parse_validity"]["value"] == 1.0      # "m_e" parses, is wrong
    assert scored["mass_dimension_validity"]["value"] == 0.0


def test_G14_symbolic_beats_raw_on_reordered_equals():
    """A correct answer written differently must not be marked wrong."""
    reference = to_prefix(to_sympy("m_e^2 + s_12"))
    reordered = ["+"] + to_prefix(to_sympy("s_12")) + to_prefix(to_sympy("m_e^2"))
    scored = evaluate_predictions([reordered], [reference])
    assert scored["sequence_exact_match"]["value"] == 0.0
    assert scored["symbolic_exact_match"]["value"] == 1.0


def test_G14_wilson_interval_brackets_the_estimate():
    p, lo, hi = wilson(34, 36)
    assert lo < p < hi and hi <= 1.0
    p0, lo0, hi0 = wilson(0, 36)
    assert p0 == 0.0 and lo0 == 0.0 and hi0 > 0.0


# --- constrained decoding ----------------------------------------------------

def test_constrained_decoding_only_emits_parseable_sequences():
    cfg, b = bundle()
    torch.manual_seed(0)
    model = AmplitudeModel(cfg.model, b.graph_vocab, b.amp_vocab,
                           b.target_vocab, b.lengths).eval()

    batch = next(iter(b.loaders["val"]))
    decoded = beam_search(model, batch, b.target_vocab, beam_width=2,
                          max_len=60, constrained=True,
                          constraint=ConstraintMask(b.target_vocab))
    # An untrained model produces nonsense, but nonsense that parses.
    parseable = sum(is_well_formed(b.target_vocab.decode(ids[1:]))
                    for ids in decoded)
    assert parseable >= len(decoded) - 1, (
        f"only {parseable}/{len(decoded)} constrained outputs parse")


def test_encoder_depths_are_independent():
    """Regression: both encoders were built with cfg.graph_layers, so
    math_layers was silently dead config and the math stack took the graph
    depth."""
    cfg, b = bundle()
    model_cfg = cfg.model.__class__(**{**vars(cfg.model),
                                       "graph_layers": 1, "math_layers": 3})
    model = AmplitudeModel(model_cfg, b.graph_vocab, b.amp_vocab,
                           b.target_vocab, b.lengths,
                           segment_amp=b.segment_amp,
                           segment_len=b.segment_len)
    assert len(model.graph_encoder.blocks) == 1
    assert len(model.math_encoder.blocks) == 3


def test_checkpoint_round_trip_reproduces_predictions():
    """A checkpoint must be enough to rebuild the model and decode identically.

    Training used to keep the selected weights in memory only, so a finished
    run left nothing to run inference from.
    """
    import tempfile
    from symba.checkpoint import load as load_checkpoint
    from symba.checkpoint import save as save_checkpoint

    cfg, b = bundle()
    torch.manual_seed(0)
    model = AmplitudeModel(cfg.model, b.graph_vocab, b.amp_vocab,
                           b.target_vocab, b.lengths,
                           segment_amp=b.segment_amp,
                           segment_len=b.segment_len).eval()

    batch = next(iter(b.loaders["val"]))
    constraint = ConstraintMask(b.target_vocab)
    before = beam_search(model, batch, b.target_vocab, beam_width=2,
                         max_len=40, constrained=True, constraint=constraint)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        save_checkpoint(path, model, cfg, b)
        restored, cfg2, vocabs, meta = load_checkpoint(path)

    assert vocabs[2].itos == b.target_vocab.itos
    assert meta["segment_amp"] == b.segment_amp
    assert cfg2.model.d_model == cfg.model.d_model

    after = beam_search(restored, batch, vocabs[2], beam_width=2, max_len=40,
                        constrained=True,
                        constraint=ConstraintMask(vocabs[2]))
    assert before == after, "reloaded checkpoint decodes differently"


def test_inference_path_matches_training_preprocessing():
    """The predictor must build the same tensors the trainer did."""
    from symba.inference import prepare_record

    _cfg, b = bundle()
    record = b.test[0]
    prepared = prepare_record(record.interaction, record.vertices, record.amp)

    assert prepared["graph_tokens"] == record["graph_tokens"]
    assert prepared["amp_tokens"] == record["amp_tokens"]
    assert prepared["amp_segments"] == record["amp_segments"] or \
        not b.segment_amp


def test_segmentation_preserves_the_amplitude():
    """Splitting into diagrams must not lose or invent content.

    Every token of the flat prefix has to reappear across the segments, except
    the ``+`` operators that joined the diagrams, which the split replaces with
    a single ``<diagrams>`` placeholder in the context segment.
    """
    from collections import Counter
    from symba.data.ast_parse import amp_to_segments

    _cfg, b = bundle("QCD")
    multi = 0
    for record in b.records[:40]:
        flat = Counter(record["amp_tokens"])
        segments = amp_to_segments(record["amp_std"])
        seen = Counter(t for s in segments for t in s)
        n_diagrams = len(segments) - 1
        multi += n_diagrams > 1

        seen["<diagrams>"] -= 1
        assert not (seen - flat), "segmentation invented tokens"
        missing = flat - seen
        assert sum(missing.values()) <= max(0, n_diagrams), \
            f"segmentation lost {sum(missing.values())} tokens"

    assert multi > 0, "no QCD amplitude split into diagrams at all"


def test_segmentation_is_decided_by_measurement():
    """The auto setting must pick per corpus, not per hardcoded theory.

    QCD amplitudes are long and gain 10.7x less attention work; QED's are
    already short, and the rectangular segment padding costs more than it
    saves. Measured epoch times: QCD 107s -> 42s, QED 13s -> 31s.
    """
    from symba.data.pipeline import resolve_segmentation

    _cfg, qed = bundle("QED")
    _cfg2, qcd = bundle("QCD")
    assert qcd.segment_amp is True
    assert qed.segment_amp is False

    assert resolve_segmentation(True, qed.records) is True
    assert resolve_segmentation(False, qcd.records) is False


def test_model_runs_under_both_segmentation_settings():
    for theory in ("QED", "QCD"):
        cfg, b = bundle(theory)
        batch = next(iter(b.loaders["val"]))
        for segment in (True, False):
            model = AmplitudeModel(cfg.model, b.graph_vocab, b.amp_vocab,
                                   b.target_vocab, b.lengths,
                                   segment_amp=segment).eval()
            with torch.no_grad():
                out = model(batch)
            assert out["logits"].shape[:2] == (batch["target"].size(0),
                                               batch["target"].size(1) - 1)


def test_kv_cache_matches_full_recompute():
    """Incremental decoding must be an optimisation, not a change of model.

    The cache has to travel with the beam when hypotheses are reordered, and
    the positional embedding has to see the true index, or the speedup is
    silently a different model.
    """
    cfg, b = bundle()
    batch = next(iter(b.loaders["val"]))

    for kind in ("vanilla", "xsa_proj", "xsa_mask"):
        torch.manual_seed(0)
        model = AmplitudeModel(
            cfg.model.__class__(**{**vars(cfg.model), "attention": kind}),
            b.graph_vocab, b.amp_vocab, b.target_vocab, b.lengths).eval()

        with torch.no_grad():
            memory, mask, _ = model.encode(batch)
            seq = batch["target"][:, :12]
            full, _ = model.decode_step(seq, memory, mask)

            cache, stepwise = model.new_cache(), None
            for t in range(seq.size(1)):
                stepwise, _ = model.decode_step(seq[:, t:t + 1], memory, mask,
                                                cache=cache, offset=t)

        diff = (full[:, -1] - stepwise[:, -1]).abs().max().item()
        assert diff < 1e-4, f"{kind}: cached logits differ by {diff:.2e}"


def test_beam_search_is_identical_with_and_without_cache():
    cfg, b = bundle()
    torch.manual_seed(0)
    model = AmplitudeModel(cfg.model, b.graph_vocab, b.amp_vocab,
                           b.target_vocab, b.lengths).eval()
    batch = next(iter(b.loaders["val"]))
    constraint = ConstraintMask(b.target_vocab)

    kwargs = dict(beam_width=4, max_len=60, constrained=True,
                  constraint=constraint)
    cached = beam_search(model, batch, b.target_vocab, use_cache=True, **kwargs)
    plain = beam_search(model, batch, b.target_vocab, use_cache=False, **kwargs)
    assert cached == plain


def test_prefix_state_tracks_operand_debt():
    state = PrefixState()
    for token in ["+", "m_e"]:
        state = state.advance(token)
    assert not state.is_complete()
    state = state.advance("s_12")
    assert state.is_complete()


def test_prefix_state_counts_an_integer_as_one_operand():
    """An integer must settle the same debt a symbol does.

    Getting this wrong makes the state machine think the expression never
    closes, so the length-budget constraint never engages and the decoder runs
    to max_len emitting an unparseable prefix.
    """
    state = PrefixState()
    for token in ["+", "INT+", "1", "6"]:
        state = state.advance(token)
    assert not state.is_complete()
    state = state.advance("m_e")
    assert state.is_complete()

    # And the state machine agrees with the real parser on every target.
    _cfg, b = bundle()
    for record in b.train[:40]:
        state, tokens = PrefixState(), record["target_tokens"]
        for token in tokens:
            state = state.advance(token)
        assert state.is_complete(), tokens[:12]


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
