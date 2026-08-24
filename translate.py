"""
Interactive English -> Spanish translator.

    python translate.py                                  # interactive prompt
    python translate.py --text "Where is the station?"   # one sentence
    python translate.py --file in.txt --out es.txt       # batch a file
    python translate.py --compare --text "..."           # greedy vs beam

Runs on CPU. A checkpoint is required: either the training checkpoint
(`best.pt`) or a slimmed one from `export_model.py`.

Expected quality: this model scores ~26 BLEU on OPUS-100 test data. It is
trained on 100k subtitle sentence pairs, so it handles short conversational
English reasonably and degrades on long sentences, idioms, technical register,
and proper nouns. It is a course project, not a production system.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from config import ARTIFACT_DIR, CFG, get_model_config
from decode import translate_corpus
from model import TransformerNMT
from tokenizer import NMTTokenizer

# Tuned on validation per preset (Section 4.5 of the report). The BLEU-optimal
# length ratio is ~0.98, so these deliberately do not target parity.
DEFAULT_ALPHA = {"base": 2.6, "small": 2.2}


def find_checkpoint(preset: str, explicit: str | None = None) -> Path | None:
    """Look in the places a checkpoint plausibly lands.

    Downloading from Drive tends to drop files in whatever folder the browser
    chose, so searching a few conventional locations avoids making the user
    reconstruct an exact path.
    """
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    candidates = [
        ROOT / "models" / f"{preset}_inference.pt",
        ROOT / "models" / preset / "best.pt",
        ROOT / "models" / "best.pt",
        Path(CFG.train.checkpoint_dir) / preset / f"{preset}_inference.pt",
        Path(CFG.train.checkpoint_dir) / preset / "best.pt",
        Path(CFG.train.checkpoint_dir) / preset / "last.pt",
        ROOT / f"{preset}_inference.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load(preset: str, ckpt_path: Path, device):
    tok = NMTTokenizer(ARTIFACT_DIR / f"{CFG.tok.model_prefix}.model", CFG.tok)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Prefer the config stored with the checkpoint; fall back to the preset name.
    saved = state.get("model_config")
    if isinstance(saved, dict):
        mcfg = get_model_config(preset)
        for k, v in saved.items():
            if hasattr(mcfg, k):
                setattr(mcfg, k, v)
    else:
        mcfg = get_model_config(preset)

    weights = state["model"] if "model" in state else state

    # A checkpoint trained against a different tokenizer loads without error and
    # then emits fluent nonsense, so check the embedding matrix explicitly.
    ckpt_vocab = weights["embed.weight"].shape[0]
    if ckpt_vocab != tok.vocab_size:
        raise SystemExit(
            f"Vocabulary mismatch: checkpoint expects {ckpt_vocab:,} tokens but "
            f"the tokenizer has {tok.vocab_size:,}.\n"
            f"The checkpoint and artifacts/{CFG.tok.model_prefix}.model come from "
            f"different runs. Rebuild the tokenizer with "
            f"`python run_milestone1.py --force`, or use the matching checkpoint.")

    model = TransformerNMT(tok.vocab_size, tok.pad_id, mcfg).to(device)
    model.load_state_dict(weights)
    model.eval()
    n = sum({id(p): p.numel() for p in model.parameters()}.values())
    return model, tok, mcfg, n


def translate(model, tok, sentences, device, method, alpha, beam_size):
    return translate_corpus(
        model, tok, sentences, device, method=method,
        batch_size=CFG.decode.batch_size, max_len=CFG.decode.max_len,
        beam_size=beam_size, length_penalty=alpha, progress=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="EN->ES neural machine translation")
    ap.add_argument("--text", help="translate a single sentence and exit")
    ap.add_argument("--file", help="translate a file, one sentence per line")
    ap.add_argument("--out", help="write file-mode output here (default: stdout)")
    ap.add_argument("--preset", default="base", choices=["base", "small"])
    ap.add_argument("--checkpoint", default=None, help="path to a .pt checkpoint")
    ap.add_argument("--method", default="beam", choices=["greedy", "beam"])
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=None,
                    help="length penalty (default: tuned value for the preset)")
    ap.add_argument("--compare", action="store_true",
                    help="show greedy and beam side by side")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alpha = args.alpha if args.alpha is not None else DEFAULT_ALPHA[args.preset]

    ckpt = find_checkpoint(args.preset, args.checkpoint)
    if ckpt is None:
        sys.exit(
            f"No {args.preset} checkpoint found.\n\n"
            f"Put one at:  models/{args.preset}_inference.pt\n\n"
            f"To produce it, run this in Colab after training:\n"
            f"    !python export_model.py --preset {args.preset} "
            f'--checkpoint-dir "{{CKPT_DIR}}"\n'
            f"then download {args.preset}_inference.pt from Drive.\n\n"
            f"Run `python setup_local.py` to check your setup.")

    spm = ARTIFACT_DIR / f"{CFG.tok.model_prefix}.model"
    if not spm.exists():
        sys.exit(f"No tokenizer at {spm}\n"
                 f"It is tracked in the repo; if missing, run "
                 f"`python run_milestone1.py --force` to rebuild it.")

    model, tok, mcfg, n_params = load(args.preset, ckpt, device)

    run = lambda sents, method, a: translate(model, tok, sents, device, method,
                                             a, args.beam_size)

    # ---------------- single sentence ----------------
    if args.text:
        if args.compare:
            g = run([args.text], "greedy", alpha)[0]
            b = run([args.text], "beam", alpha)[0]
            print(f"EN      {args.text}")
            print(f"greedy  {g}")
            print(f"beam    {b}")
        else:
            print(run([args.text], args.method, alpha)[0])
        return

    # ---------------- file ----------------
    if args.file:
        src_path = Path(args.file)
        if not src_path.exists():
            sys.exit(f"File not found: {src_path}")
        sents = [l.strip() for l in src_path.read_text(encoding="utf-8").splitlines()
                 if l.strip()]
        print(f"translating {len(sents):,} lines ({args.method}) ...", file=sys.stderr)
        t0 = time.time()
        hyps = run(sents, args.method, alpha)
        dt = time.time() - t0
        print(f"done in {dt:.1f}s ({len(sents) / max(dt, 1e-9):.1f} lines/s)",
              file=sys.stderr)
        if args.out:
            Path(args.out).write_text("\n".join(hyps) + "\n", encoding="utf-8")
            print(f"wrote {args.out}", file=sys.stderr)
        else:
            print("\n".join(hyps))
        return

    # ---------------- interactive ----------------
    print("=" * 62)
    print("  English -> Spanish  |  Transformer NMT")
    print("=" * 62)
    print(f"  model      {args.preset} ({n_params:,} params, "
          f"{mcfg.n_encoder_layers}+{mcfg.n_decoder_layers} layers)")
    print(f"  decoding   {args.method}"
          + (f" k={args.beam_size}, alpha={alpha}" if args.method == "beam" else ""))
    print(f"  device     {device}")
    print()
    print("  Type an English sentence and press Enter.")
    print("  Commands:  :greedy   :beam   :compare   :alpha <x>   :quit")
    print("=" * 62)

    method, compare = args.method, args.compare
    while True:
        try:
            line = input("\nEN> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return
        if not line:
            continue

        if line.startswith(":"):
            cmd, *rest = line[1:].split()
            cmd = cmd.lower()
            if cmd in ("quit", "q", "exit"):
                print("bye")
                return
            if cmd == "greedy":
                method, compare = "greedy", False
                print("  -> greedy decoding")
            elif cmd == "beam":
                method, compare = "beam", False
                print(f"  -> beam search k={args.beam_size}, alpha={alpha}")
            elif cmd == "compare":
                compare = True
                print("  -> showing both")
            elif cmd == "alpha" and rest:
                try:
                    alpha = float(rest[0])
                    print(f"  -> length penalty alpha={alpha}")
                except ValueError:
                    print("  usage: :alpha 2.0")
            else:
                print("  commands: :greedy :beam :compare :alpha <x> :quit")
            continue

        t0 = time.time()
        if compare:
            g = run([line], "greedy", alpha)[0]
            b = run([line], "beam", alpha)[0]
            print(f"ES> greedy  {g}")
            print(f"ES> beam    {b}")
        else:
            print(f"ES> {run([line], method, alpha)[0]}")
        print(f"    ({time.time() - t0:.2f}s)")


if __name__ == "__main__":
    main()
