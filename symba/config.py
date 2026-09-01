"""Experiment configuration: one dataclass tree.

Every knob the pipeline reads lives here, and nothing downstream defines its
own default, so a run is fully described by this config plus its seed. Named
combinations live in ``symba/experiment.py`` as ARMS.
"""

from dataclasses import asdict, dataclass, field, replace

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
    # record = protocol A, template = protocol B (01 SS5). Protocol C
    # (cross-theory transfer) is specified in the docs but not implemented;
    # split_records raises on anything else rather than silently falling back.
    split_protocol: str = "record"      # record | template
    val_frac: float = 0.1
    test_frac: float = 0.1
    target: str = "canonical"           # canonical | raw
    amp_representation: str = "ast"     # ast | raw
    # Encode each Feynman diagram of the amplitude separately with shared
    # weights instead of as one flat stream (01 SS4.2).
    #
    # "auto" measures whether it pays on this corpus and decides. It does not
    # always: segments are batched as a rectangle (n_diagrams x longest
    # diagram), so a theory whose amplitudes are already short loses more to
    # that padding than it saves on attention. Measured, per training epoch:
    #   QCD  107s flat -> 42s segmented
    #   QED   13s flat -> 31s segmented
    # True | False force it, for the ablation arm.
    segment_amp: str = "auto"


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

    use_graph: bool = True
    use_math: bool = True

    # Positional table sizes come from the measured data lengths, not from a
    # 2048 default that would allocate ~3M rows never touched by a gradient
    # (02 SS3.10). There is deliberately no max_seq_len field.


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
    # Selection is on free-running symbolic exact match on the validation
    # split, never on a smoothed likelihood (02 SS5.1: the CE floor was 0.8778
    # and the run reached 0.8828, so the selection signal was 5e-3 wide).
    eval_every: int = 2                 # free-running eval cadence, in epochs
    beam_width: int = 4          # final test decoding
    # Validation evals only have to *rank* checkpoints, so they decode greedily:
    # beam search costs beam_width times more for a selection signal that barely
    # moves. The final test number still uses beam_width.
    select_beam_width: int = 1
    length_penalty: float = 0.7         # GNMT alpha; 0 disables
    constrained_decoding: bool = True


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
