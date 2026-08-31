"""Experiment configuration. One dataclass tree, one YAML per experiment.

Every knob the pipeline reads lives here. Nothing downstream defines its own
default, so a run is fully described by its config plus its seed.
"""

from dataclasses import dataclass, field, asdict, replace
from typing import Optional

import json
import os

# Fixed vocabulary slots. Digits 0-9 occupy ids 4..13 (see data/vocab.py).
PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

# Mass dimensions used by the homogeneity check (01 §0.3).
MASS_DIMENSION = {"mass": 1, "mandelstam": 2, "reg_prop": 2, "coupling": 0}

# Token type channel (01 §S5). Order fixes the embedding row ids.
TOKEN_TYPES = ["SPECIAL", "OP", "DIGIT", "MASS", "MANDELSTAM",
               "COUPLING", "REGPROP", "STRUCT"]
TYPE_TO_ID = {t: i for i, t in enumerate(TOKEN_TYPES)}


@dataclass
class DataConfig:
    root: str = "data/Symba"
    theory: str = "QED"                 # QED | QCD
    split_protocol: str = "record"      # record (A) | template (B) | cross (C)
    val_frac: float = 0.1
    test_frac: float = 0.1
    target: str = "canonical"           # canonical | raw
    amp_representation: str = "ast"     # ast | raw
    # A parse failure is a build failure (01 P2). Only flip this to inspect a
    # broken corpus; it is asserted off in the gate tests.
    allow_parse_failures: bool = False


@dataclass
class ModelConfig:
    d_model: int = 128
    num_heads: int = 4
    graph_layers: int = 2
    math_layers: int = 2
    dec_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    attn_dropout: float = 0.1

    attention: str = "vanilla"          # vanilla | xsa_proj | xsa_mask
    ffn: str = "dense"                  # dense | moe
    n_experts: int = 4
    top_k: int = 2
    moe_aux_weight: float = 0.01

    embedding: str = "plain"            # plain | role_filler | tpr
    use_type_embedding: bool = True
    tie_embeddings: bool = True
    tokenizer: str = "vocab"            # vocab | gbst
    gbst_block_sizes: tuple = (1, 2, 3, 4)
    gbst_downsample: int = 2

    use_graph: bool = True
    use_math: bool = True
    gated_fusion: bool = False

    # Positional tables are allocated from the fitted data lengths, never from a
    # 2048 default that would waste ~3M untouched parameters (02 §3.10).
    max_seq_len: int = 0                # 0 => derived from the dataset


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    label_smoothing: float = 0.0        # deterministic map; smoothing is opt-in
    grad_clip: float = 1.0
    batch_size: int = 16
    num_epochs: int = 60
    patience: int = 12
    seed: int = 0
    # Model selection is on a task metric, never on a smoothed likelihood
    # (02 §5.1: the CE floor was 0.8778 and the run reached 0.8828).
    select_on: str = "val_symbolic_em"
    eval_every: int = 2                 # free-running eval cadence, in epochs
    beam_width: int = 4
    length_penalty: float = 0.7         # GNMT alpha; 0 disables
    constrained_decoding: bool = True
    device: str = "auto"


@dataclass
class Config:
    name: str = "default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    out_dir: str = "results"

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        return cls(
            name=d.get("name", "default"),
            data=DataConfig(**d.get("data", {})),
            model=ModelConfig(**d.get("model", {})),
            train=TrainConfig(**d.get("train", {})),
            out_dir=d.get("out_dir", "results"),
        )

    def with_overrides(self, **kw) -> "Config":
        """Shallow override by dotted key, e.g. ``model.attention='xsa_mask'``."""
        data, model, train = self.data, self.model, self.train
        top = {}
        for key, value in kw.items():
            section, _, leaf = key.partition(".")
            if not leaf:
                top[section] = value
            elif section == "data":
                data = replace(data, **{leaf: value})
            elif section == "model":
                model = replace(model, **{leaf: value})
            elif section == "train":
                train = replace(train, **{leaf: value})
            else:
                raise KeyError(f"unknown config section {section!r}")
        return replace(self, data=data, model=model, train=train, **top)
