"""Configuration for one training run."""

from dataclasses import asdict, dataclass, field

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]

# Grammar type of every token, fed to the model as an extra embedding.
TOKEN_TYPES = ["SPECIAL", "OP", "DIGIT", "MASS", "MANDELSTAM",
               "COUPLING", "REGPROP", "STRUCT"]
TYPE_TO_ID = {t: i for i, t in enumerate(TOKEN_TYPES)}


@dataclass
class DataConfig:
    root: str = "data/Symba"
    theory: str = "QED"                 # QED | QCD
    val_frac: float = 0.1
    test_frac: float = 0.1
    # Encode the amplitude one Feynman diagram at a time. None decides from
    # the data: it pays off on QCD, whose amplitudes are long, not on QED.
    segment_amp: bool = None


@dataclass
class ModelConfig:
    d_model: int = 128
    num_heads: int = 4
    graph_layers: int = 2
    math_layers: int = 2
    dec_layers: int = 2
    dim_feedforward: int = 512
    dropout: float = 0.1
    embedding: str = "role_filler"      # role_filler | plain
    attention: str = "xsa"              # xsa | vanilla (encoder self-attention)
    ffn: str = "moe"                    # moe | dense
    n_experts: int = 4
    top_k: int = 2
    moe_aux_weight: float = 0.01
    use_graph: bool = True
    use_math: bool = True


@dataclass
class TrainConfig:
    batch_size: int = 16
    epochs: int = 120
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 5                 # epochs between greedy decodes of val
    patience: int = 30                  # epochs without improvement before stopping
    beam_width: int = 4
    length_penalty: float = 0.7
    seed: int = 42


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(DataConfig(**d["data"]), ModelConfig(**d["model"]),
                   TrainConfig(**d["train"]))
