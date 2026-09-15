"""Checks on preprocessing, decoding and the model.

    python tests/test_pipeline.py
"""

import dataclasses
import os
import sys
import tempfile

import sympy
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from symba.config import Config, SPECIAL_TOKENS
from symba.data.canonical import canonicalise, to_sympy
from symba.data.dataset import build, collate, encode, split
from symba.data.load import load_theory
from symba.data.normalize import standardize
from symba.data.serialize import DIGITS, from_prefix, is_well_formed, target_tokens
from symba.decode import beam_search
from symba.metrics import symbolically_equal
from symba.model.model import AmplitudeModel
from symba.predictor import Predictor, save_checkpoint

DATA_ROOT = os.path.join(ROOT, "data", "Symba")
_data = {}


def tiny(**model):
    cfg = Config()
    cfg.data.root = DATA_ROOT
    cfg.model = dataclasses.replace(cfg.model, d_model=32, num_heads=2, dim_feedforward=64, **model)
    cfg.train.batch_size = 8
    return cfg


def qed():
    if "qed" not in _data:
        _data["qed"] = build(tiny(), verbose=False)
    return _data["qed"]


def model_for(data, **overrides):
    torch.manual_seed(0)
    return AmplitudeModel(tiny(**overrides).model, data.graph_vocab, data.amp_vocab,
                          data.target_vocab, data.table_sizes)


def test_dummy_indices_are_renumbered():
    raw = r"gamma_{%\sigma_18923}*delta_{%\mu_99}*gamma_{%\sigma_18923}*s_12"
    out = standardize(raw)
    assert out == r"gamma_{%\sigma_d1}*delta_{%\mu_d2}*gamma_{%\sigma_d1}*s_12"
    assert standardize(out) == out


def test_canonical_target_is_the_same_function_and_round_trips():
    for theory in ("QED", "QCD"):
        for record in load_theory(DATA_ROOT, theory)[::25]:
            sq_amp = standardize(record["sq_amp"])
            numerator, denominator = canonicalise(sq_amp)
            assert sympy.cancel(to_sympy(sq_amp) - numerator / denominator) == 0
            tokens = target_tokens(numerator, denominator)
            assert is_well_formed(tokens)
            assert sympy.cancel(from_prefix(tokens) - numerator / denominator) == 0


def test_split_is_disjoint_and_deterministic():
    items = list(range(100))
    parts = split(items, 0.1, 0.1, seed=42)
    assert parts == split(items, 0.1, 0.1, seed=42)
    assert sorted(parts[0] + parts[1] + parts[2]) == items
    assert [len(p) for p in parts] == [80, 10, 10]


def test_target_vocabulary_comes_from_the_training_split():
    data = qed()
    seen = {t for r in data.train for t in r["target_tokens"]}
    assert set(data.target_vocab.itos) - set(SPECIAL_TOKENS) - set(DIGITS) <= seen


def test_symbolic_equality():
    ref = ["/", "s_12", "+", "s_13", "m_e"]
    assert symbolically_equal(["/", "s_12", "+", "m_e", "s_13"], ref)
    assert not symbolically_equal(["/", "s_12", "+", "s_14", "m_e"], ref)
    assert not symbolically_equal(["/", "s_12", "+", "s_13"], ref)


def test_every_variant_trains_and_decodes_valid_expressions():
    data = qed()
    batch = next(iter(data.loaders["val"]))
    for overrides in ({}, {"use_math": False}, {"use_graph": False},
                      {"embedding": "plain"}, {"ffn": "dense"}, {"attention": "vanilla"}):
        model = model_for(data, **overrides)
        out = model(batch)
        assert out["logits"].shape[:2] == batch["target"][:, 1:].shape
        out["logits"].sum().backward()
        for ids in beam_search(model, batch, data.target_vocab, 2, data.decode_budget):
            assert is_well_formed(data.target_vocab.decode(ids[1:]))


def test_kv_cache_does_not_change_decoding():
    data = qed()
    batch = next(iter(data.loaders["val"]))
    model = model_for(data)
    args = (model, batch, data.target_vocab, 3, data.decode_budget)
    assert beam_search(*args, use_cache=True) == beam_search(*args, use_cache=False)


def test_checkpoint_reproduces_predictions():
    data = qed()
    cfg = tiny()
    cfg.data.segment_amp = data.segment_amp
    model = model_for(data)
    record = data.test[0]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "model.pt")
        save_checkpoint(path, model, cfg, data)
        _, tokens = Predictor(path).predict(record["interaction"], record["vertices"],
                                            record["amp"])
    batch = collate([encode(record, data.graph_vocab, data.amp_vocab)])
    ids = beam_search(model, batch, data.target_vocab, cfg.train.beam_width,
                      data.decode_budget, cfg.train.length_penalty)[0]
    assert tokens == data.target_vocab.decode(ids[1:])


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} passed")
