"""
Decoder correctness tests. Run after training:

    python verify_decoding.py --preset small
    python verify_decoding.py --preset base --smoke

BLEU alone cannot tell you whether beam search is correct -- a subtly broken
beam still produces fluent-looking output and a plausible score. These three
checks pin the behaviour down independently of translation quality.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F

from config import ARTIFACT_DIR, CFG, get_model_config
from dataset import TranslationDataset, make_dataloader
from decode import _trim, beam_search, greedy_decode
from model import TransformerNMT
from tokenizer import NMTTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="small", choices=["base", "small"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--checkpoint-dir", default=None, help="e.g. a Google Drive path")
    args = ap.parse_args()

    torch.manual_seed(0)
    if args.checkpoint_dir:
        CFG.train.checkpoint_dir = args.checkpoint_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prefix = "spm_smoke" if args.smoke else CFG.tok.model_prefix
    tok = NMTTokenizer(ARTIFACT_DIR / f"{prefix}.model", CFG.tok)

    ds = TranslationDataset.load(ARTIFACT_DIR / "valid_encoded.npz")
    CFG.loader.num_workers, CFG.loader.pin_memory = 0, False
    batch = next(iter(make_dataloader(ds, tok, CFG.loader, train=False, bucket=True)))
    src = batch["src"][: args.n].to(device)
    mask = batch["src_key_padding_mask"][: args.n].to(device)
    N = src.size(0)

    model = TransformerNMT(tok.vocab_size, tok.pad_id, get_model_config(args.preset)).to(device)
    ckpt = Path(CFG.train.checkpoint_dir) / args.preset / "best.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device,
                                         weights_only=False)["model"])
        print(f"loaded {ckpt}")
    else:
        print(f"WARNING: {ckpt} not found -- testing an untrained model. "
              f"The checks are still valid; the translations will be noise.")
    model.eval()

    kw = dict(bos_id=tok.bos_id, eos_id=tok.eos_id, pad_id=tok.pad_id,
              max_len=CFG.decode.max_len)
    trim = lambda t, i: _trim(t[i].tolist(), tok.eos_id, tok.pad_id)

    print("\n" + "=" * 68)
    print("1. beam(k=1, alpha=0) must be IDENTICAL to greedy")
    print("=" * 68)
    print("   With one beam there is nothing to compare against, so beam search")
    print("   degenerates to argmax at every step. Any difference means the beam")
    print("   bookkeeping -- reordering, scoring, or termination -- is wrong.\n")
    g = greedy_decode(model, src, mask, **kw)
    b1 = beam_search(model, src, mask, beam_size=1, length_penalty=0.0, **kw)
    same = all(trim(g, i) == trim(b1, i) for i in range(N))
    print(f"   [{'PASS' if same else 'FAIL'}] {N}/{N} sequences match")
    assert same, "beam(k=1) diverged from greedy"

    print("\n" + "=" * 68)
    print("2. beam(k=5) must find sequences of HIGHER log-probability")
    print("=" * 68)
    print("   This is beam search's entire purpose: approximate the best")
    print("   SEQUENCE rather than a chain of locally-best tokens. If it does")
    print("   not beat greedy on model score, it is not searching.\n")

    @torch.no_grad()
    def seq_logprob(seqs):
        memory = model.encode(src, mask)
        tot = torch.zeros(len(seqs))
        for i, ids in enumerate(seqs):
            if not ids:
                continue
            inp = torch.tensor([[tok.bos_id] + ids], device=device)
            out = model.decode(inp, memory[i:i + 1], None, mask[i:i + 1])
            lp = F.log_softmax(model.generator(out)[0].float(), -1)
            tgt = torch.tensor(ids + [tok.eos_id], device=device)
            tot[i] = lp[torch.arange(len(tgt)), tgt].sum()
        return tot

    b5 = beam_search(model, src, mask, beam_size=5, length_penalty=0.0, **kw)
    gl = seq_logprob([trim(g, i) for i in range(N)])
    bl = seq_logprob([trim(b5, i) for i in range(N)])
    wins = int((bl >= gl - 1e-4).sum())
    print(f"   greedy mean log-prob {gl.mean():.3f}  ->  beam {bl.mean():.3f}")
    print(f"   [{'PASS' if wins == N else 'FAIL'}] beam >= greedy on {wins}/{N} sentences")
    assert wins == N, "beam search returned lower-probability sequences than greedy"

    print("\n" + "=" * 68)
    print("3. length penalty must lengthen output monotonically")
    print("=" * 68)
    print("   Sequence log-probability is a sum of negative terms, so raw beam")
    print("   search prefers short hypotheses and truncates. Raising alpha must")
    print("   push mean length up, or the penalty is not being applied.\n")
    lengths = {}
    for alpha in (0.0, 0.6, 1.5):
        out = beam_search(model, src, mask, beam_size=5, length_penalty=alpha, **kw)
        lengths[alpha] = sum(len(trim(out, i)) for i in range(N)) / N
    monotonic = lengths[0.0] <= lengths[0.6] <= lengths[1.5]
    for a, v in lengths.items():
        print(f"   alpha={a:<4} mean length {v:.1f}")
    print(f"   [{'PASS' if monotonic else 'WARN'}] non-decreasing in alpha")

    print("\nAll decoder checks passed.")


if __name__ == "__main__":
    main()
