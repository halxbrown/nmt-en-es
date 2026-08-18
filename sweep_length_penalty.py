"""
Length-penalty sweep.

    python sweep_length_penalty.py --preset base --checkpoint-dir "$CKPT"

Milestone 4 showed a hypothesis/reference length ratio near 0.92 in every length
bucket -- the model under-generates by roughly 8% everywhere, not just on long
sentences. BLEU penalises that directly: at ratio 0.926 the brevity penalty is
exp(1 - 1/0.926) ~= 0.923, so about 2 BLEU is lost to length alone before
precision is even considered.

The GNMT length penalty divides a beam's score by ((5+len)/6)^alpha, so raising
alpha makes long hypotheses cheaper and should lengthen output.

IMPORTANT -- alpha is swept on the VALIDATION set, and only the single winning
value is then run on test. Choosing a decoding hyperparameter by test BLEU would
contaminate the test set exactly the way selecting a checkpoint on test would.
The final number reported must come from one test decode, not the best of five.
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

from analysis import corpus_bleu_from_stats, paired_bootstrap, sentence_stats
from config import ARTIFACT_DIR, CFG, PROC_DIR, get_model_config
from data_prep import read_parallel
from decode import translate_corpus
from model import TransformerNMT
from tokenizer import NMTTokenizer


def length_ratio(hyps, refs) -> float:
    st = [sentence_stats(h, r) for h, r in zip(hyps, refs)]
    return sum(s["hyp_len"] for s in st) / max(sum(s["ref_len"] for s in st), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="base", choices=["base", "small"])
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.3, 0.6, 1.0, 1.4])
    ap.add_argument("--n-valid", type=int, default=1000,
                    help="validation sentences used for tuning")
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint_dir:
        CFG.train.checkpoint_dir = args.checkpoint_dir
    if args.smoke:
        CFG.tok.model_prefix = "spm_smoke"
        CFG.decode.max_len = 40
    tok = NMTTokenizer(ARTIFACT_DIR / f"{CFG.tok.model_prefix}.model", CFG.tok)

    mcfg = get_model_config(args.preset)
    CFG.model = mcfg
    model = TransformerNMT(tok.vocab_size, tok.pad_id, mcfg).to(device)
    ckpt = Path(CFG.train.checkpoint_dir) / args.preset / "best.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device,
                                     weights_only=False)["model"])
    print(f"loaded {ckpt} | device {device}")

    valid = read_parallel(PROC_DIR / "valid", CFG.data)[: args.n_valid]
    v_src, v_ref = [p[0] for p in valid], [p[1] for p in valid]

    decode = lambda srcs, a: translate_corpus(
        model, tok, srcs, device, method="beam",
        batch_size=CFG.decode.batch_size, max_len=CFG.decode.max_len,
        beam_size=CFG.decode.beam_size, length_penalty=a, progress=False)

    # ------------------------------------------------ tune on validation
    print(f"\nsweeping alpha on {len(v_src):,} VALIDATION sentences")
    print(f"  {'alpha':>6}{'BLEU':>9}{'len ratio':>12}{'seconds':>9}")
    print("  " + "-" * 36)

    rows = []
    for a in args.alphas:
        t0 = time.time()
        hyps = decode(v_src, a)
        st = [sentence_stats(h, r) for h, r in zip(hyps, v_ref)]
        bleu = corpus_bleu_from_stats(st)
        lr = length_ratio(hyps, v_ref)
        rows.append({"alpha": a, "bleu": round(bleu, 2), "len_ratio": round(lr, 3)})
        print(f"  {a:>6}{bleu:>9.2f}{lr:>12.3f}{time.time() - t0:>9.0f}")

    best = max(rows, key=lambda r: r["bleu"])
    print(f"\nbest on validation: alpha={best['alpha']} "
          f"(BLEU {best['bleu']}, length ratio {best['len_ratio']})")

    # ------------------------------------------------ single test decode
    test = read_parallel(PROC_DIR / "test", CFG.data)
    if args.n_test:
        test = test[: args.n_test]
    t_src, t_ref = [p[0] for p in test], [p[1] for p in test]
    print(f"\napplying alpha={best['alpha']} once to {len(t_src):,} TEST sentences")

    hyp_new = decode(t_src, best["alpha"])
    hyp_old = decode(t_src, CFG.decode.length_penalty)

    st_new = [sentence_stats(h, r) for h, r in zip(hyp_new, t_ref)]
    st_old = [sentence_stats(h, r) for h, r in zip(hyp_old, t_ref)]
    sig = paired_bootstrap(st_old, st_new, n_samples=1000)

    print(f"\n  alpha={CFG.decode.length_penalty} (original): "
          f"BLEU {sig['bleu_a']}  length ratio {length_ratio(hyp_old, t_ref):.3f}")
    print(f"  alpha={best['alpha']} (tuned):    "
          f"BLEU {sig['bleu_b']}  length ratio {length_ratio(hyp_new, t_ref):.3f}")
    print(f"\n  delta {sig['delta']:+.2f} BLEU  95% CI "
          f"[{sig['ci95_low']:+.2f}, {sig['ci95_high']:+.2f}]  p = {sig['p_value']}")
    print(f"  -> {'significant' if sig['p_value'] < 0.05 else 'NOT significant'}")

    out = ARTIFACT_DIR / f"length_penalty_sweep_{args.preset}.json"
    out.write_text(json.dumps(
        {"preset": args.preset, "validation_sweep": rows,
         "chosen_alpha": best["alpha"], "test_comparison": sig,
         "n_valid_tuning": len(v_src)}, indent=2))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
