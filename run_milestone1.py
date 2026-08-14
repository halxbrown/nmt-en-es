"""
Milestone 1 driver.

    python run_milestone1.py              # full run against opus-100
    python run_milestone1.py --smoke      # tiny synthetic corpus, no download
    python run_milestone1.py --force      # rebuild data + tokenizer from scratch

Prints a sanity report and writes artifacts/milestone1_stats.json, which is the
evidence you cite in the report's Experimental Setup section.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from config import ARTIFACT_DIR, CFG, PROC_DIR
from data_prep import prepare, read_parallel, write_parallel
from dataset import (TranslationDataset, encode_corpus, make_dataloader,
                         padding_efficiency)
from masking import verify_masks
from tokenizer import NMTTokenizer, tokenizer_health_report, train_sentencepiece


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def build_smoke_corpus() -> dict[str, int]:
    """Synthetic parallel data so the pipeline can be tested with no network."""
    en_words = ("the cat sat on a warm mat while rain fell over quiet streets "
                "and children walked home with heavy books").split()
    es_words = ("el gato se sento en una alfombra tibia mientras la lluvia caia "
                "sobre calles tranquilas y los ninos volvian a casa").split()
    rng = random.Random(0)
    sizes = {}
    for split, n in (("train", 4000), ("valid", 300), ("test", 300)):
        pairs = []
        for _ in range(n):
            k = rng.randint(3, 25)
            pairs.append((" ".join(rng.choices(en_words, k=k)),
                          " ".join(rng.choices(es_words, k=int(k * 1.2) or 1))))
        write_parallel(pairs, PROC_DIR / split, CFG.data)
        sizes[split] = n
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="synthetic corpus, no download")
    ap.add_argument("--force", action="store_true", help="rebuild everything")
    args = ap.parse_args()

    set_seed(CFG.data.seed)
    stats: dict = {"config": {
        "dataset": f"{CFG.data.hf_dataset}:{CFG.data.hf_config}",
        "target_train_pairs": CFG.data.target_train_pairs,
        "vocab_size": CFG.tok.vocab_size,
        "model_type": CFG.tok.model_type,
        "batch_size": CFG.loader.batch_size,
        "max_tokens_per_side": CFG.loader.max_tokens_per_side,
        "seed": CFG.data.seed,
    }}

    # ---------------------------------------------------------------- 1. data
    rule("1. DATA PREPARATION")
    if args.smoke:
        vocab_override = 400
        CFG.tok.vocab_size = vocab_override
        CFG.tok.model_prefix = "spm_smoke"
        stats["config"]["vocab_size"] = vocab_override
        sizes = build_smoke_corpus()
    else:
        sizes = prepare(CFG.data, force=args.force)
    stats["split_sizes"] = sizes
    print("\nsplit sizes:", sizes)

    train_pairs = read_parallel(PROC_DIR / "train", CFG.data)
    valid_pairs = read_parallel(PROC_DIR / "valid", CFG.data)

    # ----------------------------------------------------------- 2. tokenizer
    rule("2. TOKENIZER")
    model_path = train_sentencepiece(CFG.tok, CFG.data.src_lang, CFG.data.tgt_lang,
                                     force=args.force or args.smoke)
    tok = NMTTokenizer(model_path, CFG.tok)
    print(f"vocab size: {tok.vocab_size:,}   "
          f"pad={tok.pad_id} unk={tok.unk_id} bos={tok.bos_id} eos={tok.eos_id}")

    print("\n" + tokenizer_health_report(tok, [p[0] for p in train_pairs[:20000]], "train.en"))
    print(tokenizer_health_report(tok, [p[1] for p in train_pairs[:20000]], "train.es"))
    print(tokenizer_health_report(tok, [p[0] for p in valid_pairs], "valid.en (unseen)"))
    print(tokenizer_health_report(tok, [p[1] for p in valid_pairs], "valid.es (unseen)"))

    print("\nround-trip check:")
    for s, t in train_pairs[:3]:
        rt = tok.decode(tok.encode_source(s))
        flag = "OK  " if rt.strip() == s.strip() else "DIFF"
        print(f"  [{flag}] {s[:58]}")
        print(f"         -> {tok.pieces(s)[:12]}")

    # -------------------------------------------------------- 3. encode/cache
    rule("3. ENCODING + CACHING")
    datasets = {}
    for split, pairs in (("train", train_pairs), ("valid", valid_pairs)):
        cache = ARTIFACT_DIR / f"{split}_encoded.npz"
        if cache.exists() and not (args.force or args.smoke):
            ds = TranslationDataset.load(cache)
            print(f"  {split}: loaded cache ({len(ds):,} pairs)")
        else:
            src, tgt = encode_corpus(pairs, tok, CFG.loader.max_tokens_per_side)
            ds = TranslationDataset(src, tgt)
            ds.save(cache)
            print(f"  {split}: encoded and cached ({len(ds):,} pairs)")
        datasets[split] = ds

    train_ds = datasets["train"]
    sl, tl = train_ds.src.lengths, train_ds.tgt.lengths
    pct = [50, 90, 95, 99, 100]
    stats["length_percentiles"] = {
        "src": {f"p{p}": int(np.percentile(sl, p)) for p in pct},
        "tgt": {f"p{p}": int(np.percentile(tl, p)) for p in pct},
    }
    print(f"\n  src subword length: mean {sl.mean():.1f}  " +
          "  ".join(f"p{p}={int(np.percentile(sl, p))}" for p in pct))
    print(f"  tgt subword length: mean {tl.mean():.1f}  " +
          "  ".join(f"p{p}={int(np.percentile(tl, p))}" for p in pct))
    cap = CFG.loader.max_tokens_per_side
    n_trunc = int((sl >= cap).sum() + (tl >= cap).sum())
    stats["truncated_at_cap"] = n_trunc
    print(f"  total training subwords: {int(sl.sum() + tl.sum()):,}")
    print(f"  sequences hitting the {cap}-token cap: {n_trunc:,} "
          f"({100 * n_trunc / (2 * len(sl)):.3f}% of all sequences)")

    # ------------------------------------------------------------ 4. dataloader
    rule("4. DATALOADER / DYNAMIC BATCHING")
    CFG.loader.num_workers = 0  # keep the sanity run single-process and debuggable
    CFG.loader.pin_memory = False
    loader = make_dataloader(train_ds, tok, CFG.loader, train=True, bucket=True)
    print(f"batches per epoch: {len(loader):,}  (batch_size={CFG.loader.batch_size})")

    batch = next(iter(loader))
    print("\ntensor shapes:")
    for k in ("src", "decoder_input", "labels", "src_key_padding_mask", "tgt_key_padding_mask"):
        v = batch[k]
        print(f"  {k:<24} {tuple(v.shape)}  {v.dtype}")
    print(f"  {'n_tgt_tokens':<24} {batch['n_tgt_tokens']:,} real target tokens in batch")

    print("\nmask verification:")
    for line in verify_masks(batch, tok.pad_id):
        print(f"  [PASS] {line}")

    print("\nreconstructed example from the batch tensors:")
    i = 0
    print(f"  src text       : {tok.decode(batch['src'][i])}")
    print(f"  tgt text       : {tok.decode(batch['decoder_input'][i])}")
    # decode() strips specials, so print raw ids to make the one-step shift visible
    print(f"  src ids        : {batch['src'][i][:10].tolist()} ...")
    print(f"  decoder_input  : {batch['decoder_input'][i][:10].tolist()} ...   "
          f"(starts with bos={tok.bos_id})")
    print(f"  labels         : {batch['labels'][i][:10].tolist()} ...   "
          f"(same tokens, shifted left by one)")

    # ------------------------------------------------- 5. padding efficiency
    rule("5. PADDING EFFICIENCY (bucketed vs naive)")
    n_probe = min(100, len(loader))
    bucketed = padding_efficiency(loader, tok.pad_id, max_batches=n_probe)
    naive_loader = make_dataloader(train_ds, tok, CFG.loader, train=True, bucket=False)
    naive = padding_efficiency(naive_loader, tok.pad_id, max_batches=n_probe)

    stats["padding_efficiency"] = {"bucketed": bucketed, "random_batching": naive}
    print(f"  random batching : {100 * naive['useful_fraction']:.1f}% useful "
          f"({100 * naive['wasted_on_padding']:.1f}% padding)")
    print(f"  length bucketing: {100 * bucketed['useful_fraction']:.1f}% useful "
          f"({100 * bucketed['wasted_on_padding']:.1f}% padding)")
    speedup = bucketed["useful_fraction"] / max(naive["useful_fraction"], 1e-9)
    print(f"  -> {speedup:.2f}x fewer wasted attention cells per epoch")

    out = ARTIFACT_DIR / "milestone1_stats.json"
    out.write_text(json.dumps(stats, indent=2))
    rule("MILESTONE 1 COMPLETE")
    print(f"stats written to {out}")


if __name__ == "__main__":
    main()
