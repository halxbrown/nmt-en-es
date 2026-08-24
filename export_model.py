"""
Strip a training checkpoint down to inference weights.

    python export_model.py --preset base --checkpoint-dir "$CKPT"

`best.pt` carries optimizer moments, scheduler state, GradScaler state and the
full epoch history so training can resume. AdamW stores two moment tensors per
parameter, so the file is roughly three times the size of the weights alone --
about 450 MB for base. Inference needs none of it.

The exported file keeps only the state dict plus the model config, which makes
it small enough to download from Drive and run locally.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch

from config import CFG, get_model_config


def mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="base", choices=["base", "small"])
    ap.add_argument("--checkpoint-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.checkpoint_dir:
        CFG.train.checkpoint_dir = args.checkpoint_dir
    src = Path(CFG.train.checkpoint_dir) / args.preset / "best.pt"
    if not src.exists():
        sys.exit(f"No checkpoint at {src}")

    dst = Path(args.out) if args.out else src.parent / f"{args.preset}_inference.pt"
    state = torch.load(src, map_location="cpu", weights_only=False)

    mcfg = get_model_config(args.preset)
    saved = state.get("model_config")
    if isinstance(saved, dict):
        for k, v in saved.items():
            if hasattr(mcfg, k):
                setattr(mcfg, k, v)

    torch.save({"model": state["model"],
                "model_config": mcfg.__dict__,
                "preset": args.preset,
                "best_val": state.get("best_val"),
                "epoch": state.get("epoch")}, dst)

    print(f"{src.name:>22}  {mb(src):8.1f} MB")
    print(f"{dst.name:>22}  {mb(dst):8.1f} MB   "
          f"({100 * (1 - mb(dst) / mb(src)):.0f}% smaller)")
    print(f"\nwrote {dst}")
    print("Download this file plus artifacts/spm_enes.model to translate locally.")


if __name__ == "__main__":
    main()
