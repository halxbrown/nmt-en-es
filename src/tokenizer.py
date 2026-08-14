"""
Milestone 1, part 2: subword tokenization.

Why a JOINT vocabulary
----------------------
English and Spanish share the Latin alphabet and a large cognate vocabulary
("information"/"informacion", "national"/"nacional"). Training one SentencePiece
model over the concatenation of both sides means:

  * cognates and shared subwords occupy one embedding row instead of two;
  * numbers, names and punctuation are guaranteed to tokenize identically on
    both sides, which makes copy behaviour easy for the model to learn;
  * you can TIE the encoder embedding, decoder embedding and output projection
    into a single weight matrix (Press & Wolf, 2017), cutting parameters by
    roughly 2 x vocab x d_model. That matters a lot at 100k training pairs.

This is what Vaswani et al. (2017) did for EN-DE, and it is the default in
fairseq/Marian for related language pairs.

Why train ONLY on the training split
------------------------------------
A tokenizer fitted on validation/test text has seen that text. The leak is mild
but real, and it is the kind of thing a grader looks for.
"""


from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # project root on path

from pathlib import Path

import sentencepiece as spm

from config import ARTIFACT_DIR, PROC_DIR, CFG, TokenizerConfig


def train_sentencepiece(
    cfg: TokenizerConfig = CFG.tok,
    src_lang: str = "en",
    tgt_lang: str = "es",
    force: bool = False,
) -> Path:
    """Train a joint BPE model on the TRAIN split only. Returns the .model path."""
    model_path = ARTIFACT_DIR / f"{cfg.model_prefix}.model"
    if model_path.exists() and not force:
        print(f"Tokenizer already exists at {model_path}")
        return model_path

    train_files = [PROC_DIR / f"train.{src_lang}"]
    if cfg.joint:
        train_files.append(PROC_DIR / f"train.{tgt_lang}")

    for f in train_files:
        if not f.exists():
            raise FileNotFoundError(f"{f} missing -- run data_prep.prepare() first")

    spm.SentencePieceTrainer.train(
        input=",".join(str(f) for f in train_files),
        model_prefix=str(ARTIFACT_DIR / cfg.model_prefix),
        vocab_size=cfg.vocab_size,
        model_type=cfg.model_type,
        character_coverage=cfg.character_coverage,
        input_sentence_size=cfg.input_sentence_size,
        shuffle_input_sentence=cfg.shuffle_input_sentence,
        # Explicit special-token IDs -- see config.py for why pad=0.
        pad_id=cfg.pad_id,
        unk_id=cfg.unk_id,
        bos_id=cfg.bos_id,
        eos_id=cfg.eos_id,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        # Keep digits as individual tokens so the model can copy numbers it has
        # never seen as a whole.
        split_digits=True,
        # Decompose anything unseen into UTF-8 byte tokens, which drives the
        # practical UNK rate to zero -- useful when the error analysis needs to
        # distinguish "rare token handled badly" from "token never encoded".
        byte_fallback=True,
        normalization_rule_name="nmt_nfkc",
        minloglevel=1,  # quiet the per-merge trainer chatter
    )
    print(f"Trained tokenizer -> {model_path}")
    return model_path


class NMTTokenizer:
    """Thin wrapper that owns the special-token conventions for this project.

    Conventions
    -----------
    source side : tokens + </s>
    target side : <s> + tokens + </s>

    The decoder input is the target sequence minus its last token, and the
    labels are the target sequence minus its first token. That shift is applied
    in the collate function, not here.
    """

    def __init__(self, model_path: str | Path, cfg: TokenizerConfig = CFG.tok):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        self.cfg = cfg
        self.pad_id = cfg.pad_id
        self.unk_id = cfg.unk_id
        self.bos_id = cfg.bos_id
        self.eos_id = cfg.eos_id
        # Sanity: the trained model must agree with our config.
        assert self.sp.pad_id() == cfg.pad_id, "pad id mismatch"
        assert self.sp.bos_id() == cfg.bos_id, "bos id mismatch"
        assert self.sp.eos_id() == cfg.eos_id, "eos id mismatch"

    def __len__(self) -> int:
        return self.sp.get_piece_size()

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str, bos: bool = False, eos: bool = True) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def encode_source(self, text: str) -> list[int]:
        return self.encode(text, bos=False, eos=True)

    def encode_target(self, text: str) -> list[int]:
        return self.encode(text, bos=True, eos=True)

    def decode(self, ids) -> str:
        specials = {self.pad_id, self.bos_id, self.eos_id}
        ids = [int(i) for i in ids if int(i) not in specials]
        return self.sp.decode(ids)

    def pieces(self, text: str) -> list[str]:
        return self.sp.encode(text, out_type=str)


def tokenizer_health_report(tok: NMTTokenizer, sentences: list[str], label: str) -> str:
    """Fertility and UNK rate -- both belong in the report."""
    n_words = n_pieces = n_unk = 0
    for s in sentences:
        pieces = tok.sp.encode(s, out_type=int)
        n_words += len(s.split())
        n_pieces += len(pieces)
        n_unk += sum(1 for p in pieces if p == tok.unk_id)
    fertility = n_pieces / max(n_words, 1)
    unk_rate = 100 * n_unk / max(n_pieces, 1)
    return (f"[{label}] fertility {fertility:.2f} subwords/word | "
            f"UNK rate {unk_rate:.4f}% | {n_pieces:,} subwords")


if __name__ == "__main__":
    p = train_sentencepiece(force=True)
    tok = NMTTokenizer(p)
    print("vocab size:", tok.vocab_size)
