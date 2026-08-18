"""
Milestone 4: final evaluation on the held-out TEST set, plus error analysis.

    python run_milestone4.py --preset both --checkpoint-dir "$CKPT"
    python run_milestone4.py --preset base --n-test 500     # quick pass

Why this cannot reuse Milestone 3's numbers: those were computed on the
validation set, which selected the checkpoint. Reporting them as final results
would be optimistically biased. Everything here runs on `test`, untouched until
now.

Outputs
  artifacts/milestone4_results.json   all metrics, significance tests, buckets
  artifacts/failures_<preset>.md      worst sentences, annotated, for the report
  artifacts/attention_<preset>_<i>.png
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

from analysis import (bleu_by_length, category_summary, corpus_bleu_from_stats,
                      cross_attention_matrix, find_failures, paired_bootstrap,
                      plot_attention, sentence_stats)
from config import ARTIFACT_DIR, CFG, PROC_DIR, get_model_config
from data_prep import read_parallel
from decode import translate_corpus
from model import TransformerNMT
from tokenizer import NMTTokenizer


def rule(t: str) -> None:
    print("\n" + "=" * 72)
    print(t)
    print("=" * 72)


def full_metrics(hyps, refs) -> dict:
    import sacrebleu
    bleu = sacrebleu.BLEU()
    res = bleu.corpus_score(hyps, [refs])          # score first...
    out = {"bleu": round(res.score, 2),
           "chrf": round(sacrebleu.corpus_chrf(hyps, [refs]).score, 2),
           # ...then the signature is well-defined. It records tokenizer,
           # smoothing and version -- quote it in the report so the number is
           # reproducible by anyone.
           "signature": str(bleu.get_signature())}
    try:
        from rouge_score import rouge_scorer
        rs = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
        r1 = [rs.score(r, h)["rouge1"].fmeasure for h, r in zip(hyps, refs)]
        rl = [rs.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]
        out["rouge1"] = round(100 * sum(r1) / len(r1), 2)
        out["rougeL"] = round(100 * sum(rl) / len(rl), 2)
    except Exception as e:                                   # noqa: BLE001
        print(f"  (rouge unavailable: {e})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="both", choices=["base", "small", "both"])
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--n-test", type=int, default=None, help="cap test sentences")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--n-attention", type=int, default=3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint_dir:
        CFG.train.checkpoint_dir = args.checkpoint_dir
    if args.smoke:
        CFG.tok.model_prefix = "spm_smoke"
        CFG.decode.max_len = 40
    tok = NMTTokenizer(ARTIFACT_DIR / f"{CFG.tok.model_prefix}.model", CFG.tok)

    test_pairs = read_parallel(PROC_DIR / "test", CFG.data)
    if args.n_test:
        test_pairs = test_pairs[: args.n_test]
    srcs = [p[0] for p in test_pairs]
    refs = [p[1] for p in test_pairs]

    rule("MILESTONE 4 -- HELD-OUT TEST SET")
    print(f"device {device} | {len(srcs):,} test sentences | "
          f"never seen during training or checkpoint selection")

    presets = ["base", "small"] if args.preset == "both" else [args.preset]
    results, all_hyps = {}, {}

    for preset in presets:
        ckpt = Path(CFG.train.checkpoint_dir) / preset / "best.pt"
        if not ckpt.exists():
            print(f"\n!! no checkpoint at {ckpt}; skipping {preset}")
            continue

        mcfg = get_model_config(preset)
        CFG.model = mcfg
        model = TransformerNMT(tok.vocab_size, tok.pad_id, mcfg).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        n_params = sum({id(p): p.numel() for p in model.parameters()}.values())

        rule(f"{preset.upper()} -- {n_params:,} params, val loss {state['best_val']:.4f}")

        entry = {"parameters": n_params, "best_val_loss": state["best_val"],
                 "epochs_trained": state["epoch"], "decoding": {}}
        stats_by_method = {}

        for method in ("greedy", "beam"):
            t0 = time.time()
            hyps = translate_corpus(
                model, tok, srcs, device, method=method,
                batch_size=CFG.decode.batch_size, max_len=CFG.decode.max_len,
                beam_size=CFG.decode.beam_size,
                length_penalty=CFG.decode.length_penalty, progress=False)
            dt = time.time() - t0

            m = full_metrics(hyps, refs)
            m["seconds"] = round(dt, 1)
            m["sents_per_sec"] = round(len(srcs) / max(dt, 1e-9), 1)
            entry["decoding"][method] = m
            stats_by_method[method] = [sentence_stats(h, r) for h, r in zip(hyps, refs)]
            all_hyps[(preset, method)] = hyps

            label = method if method == "greedy" else f"beam k={CFG.decode.beam_size}"
            print(f"  {label:<12} BLEU {m['bleu']:>6}  chrF {m['chrf']:>6}  "
                  f"ROUGE-L {m.get('rougeL', 'n/a'):>6}  ({m['sents_per_sec']}/s)")

        # ---- is the beam gain real? ----
        sig = paired_bootstrap(stats_by_method["greedy"], stats_by_method["beam"],
                               n_samples=args.bootstrap)
        entry["beam_vs_greedy"] = sig
        verdict = "significant" if sig["p_value"] < 0.05 else "NOT significant"
        print(f"\n  paired bootstrap ({sig['n_samples']} resamples):")
        print(f"    beam - greedy = {sig['delta']:+.2f} BLEU  "
              f"95% CI [{sig['ci95_low']:+.2f}, {sig['ci95_high']:+.2f}]  "
              f"p = {sig['p_value']}  -> {verdict}")

        # ---- length buckets ----
        beam_hyps = all_hyps[(preset, "beam")]
        buckets = bleu_by_length(beam_hyps, refs, srcs)
        entry["length_buckets"] = buckets
        print(f"\n  BLEU by source length (beam):")
        print(f"    {'bucket':<10}{'n':>7}{'BLEU':>8}{'len ratio':>11}")
        for b in buckets:
            print(f"    {b['bucket']:<10}{b['n']:>7}{b['bleu']:>8}{b['len_ratio']:>11}")

        # ---- error categories ----
        cats = category_summary(srcs, beam_hyps, refs)
        entry["error_categories"] = cats
        print(f"\n  heuristic error categories (beam):")
        for k, v in cats.items():
            print(f"    {k:<22}{v['n']:>6}  {v['pct']:>6}%")

        # ---- worst sentences, written out for manual selection ----
        failures = find_failures(srcs, beam_hyps, refs, top_k=15)
        entry["failures"] = failures
        lines = [f"# Worst test sentences -- {preset} (beam k={CFG.decode.beam_size})\n",
                 "Ranked by sentence chrF. Categories are heuristic hints; read the",
                 "sentences before citing them.\n"]
        for f in failures:
            lines += [f"\n## #{f['index']}  chrF {f['chrf']}  "
                      f"[{', '.join(f['tags'])}]  len_ratio {f['len_ratio']}",
                      f"- **SRC** {f['src']}",
                      f"- **REF** {f['ref']}",
                      f"- **HYP** {f['hyp']}"]
            if f["entities"]:
                lines.append(f"- untranslated: {', '.join(f['entities'])}")
        out_md = ARTIFACT_DIR / f"failures_{preset}.md"
        out_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  {len(failures)} worst sentences -> {out_md}")

        # ---- attention maps ----
        made = []
        for i in range(min(args.n_attention, len(srcs))):
            if not (4 <= len(srcs[i].split()) <= 14):
                continue
            try:
                attn, sp, tp = cross_attention_matrix(
                    model, tok, srcs[i], beam_hyps[i], device)
                p = ARTIFACT_DIR / f"attention_{preset}_{i}.png"
                plot_attention(attn, sp, tp, f"{preset}: {srcs[i][:60]}", str(p))
                made.append(str(p))
            except Exception as e:                            # noqa: BLE001
                print(f"    (attention map {i} failed: {e})")
        entry["attention_maps"] = made
        if made:
            print(f"  {len(made)} attention maps -> artifacts/")

        results[preset] = entry
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- base vs small, on identical decoding ----
    if "base" in results and "small" in results:
        rule("BASE vs SMALL (beam, paired bootstrap)")
        sa = [sentence_stats(h, r) for h, r in zip(all_hyps[("small", "beam")], refs)]
        sb = [sentence_stats(h, r) for h, r in zip(all_hyps[("base", "beam")], refs)]
        sig = paired_bootstrap(sa, sb, n_samples=args.bootstrap)
        results["base_vs_small"] = sig
        verdict = "significant" if sig["p_value"] < 0.05 else "NOT significant"
        print(f"  small {sig['bleu_a']}  ->  base {sig['bleu_b']}")
        print(f"  delta {sig['delta']:+.2f} BLEU  95% CI "
              f"[{sig['ci95_low']:+.2f}, {sig['ci95_high']:+.2f}]  "
              f"p = {sig['p_value']}  -> {verdict}")
        extra = results["base"]["parameters"] - results["small"]["parameters"]
        print(f"  cost: +{extra:,} parameters "
              f"(+{100 * extra / results['small']['parameters']:.0f}%) "
              f"for {sig['delta']:+.2f} BLEU")

    # ---- final table ----
    if results:
        rule("FINAL TEST-SET RESULTS")
        hdr = (f"{'model':<8}{'params':>12}{'greedy':>9}{'beam':>9}"
               f"{'gain':>8}{'chrF':>8}{'ROUGE-L':>9}")
        print(hdr + "\n" + "-" * len(hdr))
        for k in presets:
            if k not in results:
                continue
            v = results[k]
            g, b = v["decoding"]["greedy"], v["decoding"]["beam"]
            print(f"{k:<8}{v['parameters']:>12,}{g['bleu']:>9}{b['bleu']:>9}"
                  f"{b['bleu'] - g['bleu']:>+8.2f}{b['chrf']:>8}"
                  f"{b.get('rougeL', 0):>9}")
        print(f"\nsacreBLEU signature: "
              f"{results[presets[0]]['decoding']['beam']['signature']}")

        out = ARTIFACT_DIR / "milestone4_results.json"
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"written to {out}")


if __name__ == "__main__":
    main()
