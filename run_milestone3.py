"""
Milestone 3 driver: training loop, then greedy vs beam decoding.

    python run_milestone3.py --preset both              # train base + small
    python run_milestone3.py --preset base --epochs 20
    python run_milestone3.py --preset both --decode-only  # reuse best.pt
    python run_milestone3.py --smoke --epochs 1         # CPU wiring check

Writes artifacts/milestone3_results.json: the decoding comparison table that
goes straight into the report.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from config import ARTIFACT_DIR, CFG, PROC_DIR, get_model_config
from data_prep import read_parallel
from dataset import TranslationDataset, make_dataloader
from decode import translate_corpus
from model import TransformerNMT
from tokenizer import NMTTokenizer
from train import train


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def score(hyps: list[str], refs: list[str]) -> dict:
    """sacreBLEU plus ROUGE-L. Milestone 4 expands on this; here it is a signal
    for choosing between the two capacity presets."""
    out = {}
    try:
        import sacrebleu
        out["bleu"] = round(sacrebleu.corpus_bleu(hyps, [refs]).score, 2)
        out["chrf"] = round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2)
    except Exception as e:                                   # noqa: BLE001
        out["bleu"] = None
        print(f"  (sacrebleu unavailable: {e})")
    try:
        from rouge_score import rouge_scorer
        rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        vals = [rs.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]
        out["rougeL"] = round(100 * sum(vals) / max(len(vals), 1), 2)
    except Exception:                                        # noqa: BLE001
        out["rougeL"] = None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="both", choices=["base", "small", "both"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--n-eval", type=int, default=500, help="validation sentences to decode")
    ap.add_argument("--checkpoint-dir", default=None, help="e.g. a Google Drive path")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(CFG.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint_dir:
        CFG.train.checkpoint_dir = args.checkpoint_dir
    if args.smoke:
        CFG.train.warmup_steps = 20
        CFG.decode.max_len = 40

    prefix = "spm_smoke" if args.smoke else CFG.tok.model_prefix
    tok = NMTTokenizer(ARTIFACT_DIR / f"{prefix}.model", CFG.tok)
    train_ds = TranslationDataset.load(ARTIFACT_DIR / "train_encoded.npz")
    valid_ds = TranslationDataset.load(ARTIFACT_DIR / "valid_encoded.npz")

    if device.type == "cpu":
        CFG.loader.num_workers = 0
        CFG.loader.pin_memory = False
    train_loader = make_dataloader(train_ds, tok, CFG.loader, train=True, bucket=True)
    valid_loader = make_dataloader(valid_ds, tok, CFG.loader, train=False, bucket=True)

    valid_pairs = read_parallel(PROC_DIR / "valid", CFG.data)[: args.n_eval]
    src_sents = [p[0] for p in valid_pairs]
    refs = [p[1] for p in valid_pairs]

    presets = ["base", "small"] if args.preset == "both" else [args.preset]
    results = {}

    for preset in presets:
        rule(f"PRESET '{preset.upper()}'")
        mcfg = get_model_config(preset)
        CFG.model = mcfg                      # so train() records the right config
        model = TransformerNMT(tok.vocab_size, tok.pad_id, mcfg).to(device)
        n_params = sum({id(p): p.numel() for p in model.parameters()}.values())
        print(f"{n_params:,} parameters | layers {mcfg.n_encoder_layers}+"
              f"{mcfg.n_decoder_layers} | d_ff {mcfg.d_ff} | dropout {mcfg.dropout}")

        ckpt = Path(CFG.train.checkpoint_dir) / preset / "best.pt"
        if args.decode_only:
            if not ckpt.exists():
                print(f"  no checkpoint at {ckpt}; skipping")
                continue
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            summary = {"tag": preset, "best_val": state["best_val"],
                       "history": state.get("history", []), "best_ckpt": str(ckpt)}
            print(f"  loaded {ckpt} (val loss {state['best_val']:.4f})")
        else:
            summary = train(model, tok, train_loader, valid_loader, device,
                            CFG.train, tag=preset, epochs=args.epochs)
            if Path(summary["best_ckpt"]).exists():
                state = torch.load(summary["best_ckpt"], map_location=device,
                                   weights_only=False)
                model.load_state_dict(state["model"])   # evaluate the BEST, not the last
                print("  restored best checkpoint for decoding")

        # ---------------- greedy vs beam ----------------
        rule(f"DECODING -- {preset} ({len(src_sents)} validation sentences)")
        decoded = {}
        for method in ("greedy", "beam"):
            t0 = time.time()
            hyps = translate_corpus(
                model, tok, src_sents, device, method=method,
                batch_size=CFG.decode.batch_size, max_len=CFG.decode.max_len,
                beam_size=CFG.decode.beam_size,
                length_penalty=CFG.decode.length_penalty,
            )
            dt = time.time() - t0
            m = score(hyps, refs)
            m["seconds"] = round(dt, 1)
            m["sents_per_sec"] = round(len(src_sents) / max(dt, 1e-9), 1)
            decoded[method] = {"metrics": m, "samples": hyps[:5]}
            label = method if method == "greedy" else f"beam k={CFG.decode.beam_size}"
            print(f"  {label:<12} BLEU {m['bleu']}  chrF {m['chrf']}  "
                  f"ROUGE-L {m['rougeL']}  ({dt:.1f}s, {m['sents_per_sec']}/s)")

        print("\n  sample translations:")
        for i in range(min(3, len(src_sents))):
            print(f"    SRC    {src_sents[i]}")
            print(f"    REF    {refs[i]}")
            print(f"    greedy {decoded['greedy']['samples'][i]}")
            print(f"    beam   {decoded['beam']['samples'][i]}\n")

        results[preset] = {"parameters": n_params, "best_val_loss": summary["best_val"],
                           "epochs_run": len(summary["history"]),
                           "history": summary["history"], "decoding": decoded}

    # ---------------- comparison table ----------------
    if results:
        rule("MILESTONE 3 SUMMARY")
        hdr = f"{'model':<8}{'params':>12}{'val loss':>10}{'greedy':>9}{'beam':>9}{'gain':>8}"
        print(hdr + "\n" + "-" * len(hdr))
        for k, v in results.items():
            g = v["decoding"]["greedy"]["metrics"]["bleu"]
            b = v["decoding"]["beam"]["metrics"]["bleu"]
            gain = f"{b - g:+.2f}" if (g is not None and b is not None) else "n/a"
            print(f"{k:<8}{v['parameters']:>12,}{v['best_val_loss']:>10.4f}"
                  f"{g:>9}{b:>9}{gain:>8}")

        out = ARTIFACT_DIR / "milestone3_results.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
