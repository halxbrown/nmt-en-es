"""
Central configuration for the EN->ES Transformer NMT project.

Everything tunable lives here so the report's "Experimental Setup" section can
be written straight from this file, and so reruns are reproducible.
"""
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT / "artifacts"

for _d in (RAW_DIR, PROC_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    # --- source ---
    hf_dataset: str = "Helsinki-NLP/opus-100"
    hf_config: str = "en-es"
    src_lang: str = "en"
    tgt_lang: str = "es"

    # --- scale (project spec: 50k-100k pairs) ---
    target_train_pairs: int = 100_000

    # --- cleaning thresholds ---
    min_words: int = 1
    max_words: int = 80
    max_len_ratio: float = 2.0          # char-length ratio between the two sides
    ratio_min_chars: int = 15           # below this, ratio is too noisy to judge
    min_alpha_frac: float = 0.5         # fraction of non-space chars that are letters
    drop_identical: bool = True         # src == tgt is copy-noise, not translation
    drop_urls: bool = True

    seed: int = 1337


@dataclass
class TokenizerConfig:
    vocab_size: int = 16_000
    model_type: str = "bpe"             # "bpe" | "unigram"
    character_coverage: float = 1.0     # 1.0 is right for Latin script
    joint: bool = True                  # one shared vocab for EN+ES
    model_prefix: str = "spm_enes"

    # Explicit IDs. pad=0 makes zero-initialised tensors naturally "padding",
    # and keeps mask logic (ids == 0) readable.
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    # SentencePiece can choke on huge corpora; sample this many lines to train on.
    input_sentence_size: int = 2_000_000
    shuffle_input_sentence: bool = True


@dataclass
class LoaderConfig:
    batch_size: int = 64
    max_tokens_per_side: int = 128      # hard truncation ceiling after tokenization
    bucket_pool_factor: int = 50        # megabatch = batch_size * this, sorted by length
    num_workers: int = 2
    pin_memory: bool = True
    seed: int = 1337


@dataclass
class ModelConfig:
    """Sized for 100k pairs (~3.8M subwords), not for WMT.

    The base Transformer (6+6 layers, d_ff=2048, d_model=512) is ~65M
    parameters and will memorise a corpus this small. Sennrich & Zhang (2019)
    show that low-resource NMT wants fewer layers, a smaller vocabulary and
    heavier regularisation, so we cut depth and raise dropout instead of
    reaching for the paper's defaults.
    """
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    dropout: float = 0.2
    max_position: int = 512
    pre_norm: bool = True       # Pre-LN: stable without a long LR warmup
    tie_embeddings: bool = True  # requires a joint vocabulary
    label_smoothing: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 20
    peak_lr: float = 5e-4
    warmup_steps: int = 2000          # ~1.3 epochs at batch 64
    min_lr_ratio: float = 0.02        # cosine floor, as a fraction of peak
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.98)
    eps: float = 1e-9
    grad_clip: float = 1.0
    amp: bool = True                  # fp16 autocast; ~2x on a T4
    early_stop_patience: int = 3
    log_every: int = 200
    checkpoint_dir: str = "checkpoints"
    seed: int = 1337


@dataclass
class DecodeConfig:
    max_len: int = 128
    beam_size: int = 5
    length_penalty: float = 0.6       # GNMT alpha; 0 disables
    batch_size: int = 32


# ---------------------------------------------------------------------------
# Capacity presets. Dropout is deliberately held constant across the two so the
# ablation isolates CAPACITY rather than confounding it with regularisation.
# ---------------------------------------------------------------------------

def get_model_config(preset: str = "base") -> "ModelConfig":
    if preset == "base":                       # ~37.6M params
        return ModelConfig()
    if preset == "small":                      # ~24M params
        return ModelConfig(n_encoder_layers=3, n_decoder_layers=3, d_ff=1024)
    raise ValueError(f"unknown preset {preset!r}; expected 'base' or 'small'")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    tok: TokenizerConfig = field(default_factory=TokenizerConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)


CFG = Config()
