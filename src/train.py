"""
Milestone 3, part 2: the training loop.

Points that matter for the report's Experimental Setup section.

LEARNING RATE SCHEDULE. Linear warmup then cosine decay. The original paper's
inverse-sqrt schedule exists mainly to keep Post-LN training stable; since the
model is Pre-LN, warmup is needed only to stop Adam's second-moment estimates
being set by a handful of noisy early batches. 2000 steps is about 1.3 epochs.

LOSS NORMALISATION. Sentence-count batching plus a skewed length distribution
means the number of target tokens per batch varies several-fold. CrossEntropyLoss
with ignore_index averages over non-padding positions only, so the gradient
scale stays comparable across batches regardless of how many tokens they hold.

MIXED PRECISION. fp16 autocast roughly doubles throughput on a T4. The attention
mask uses finfo.min rather than -inf specifically so it survives fp16 without
producing NaN.

CHECKPOINT EVERY EPOCH. Colab sessions are killed without warning. Every epoch
writes `last.pt` (for resuming) and, when validation improves, `best.pt`.
Optimizer and scheduler state are included, so a resumed run continues rather
than restarting with a cold optimizer.

EARLY STOPPING on validation loss. At 37.6M parameters against 3.8M training
subwords, training loss will keep falling long after validation loss turns.
That turning point is the signal, and its epoch index belongs in the report.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # project root on path

import json
import math
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import LambdaLR

from config import CFG, TrainConfig
from model import build_loss


def build_scheduler(optimizer, cfg: TrainConfig, total_steps: int) -> LambdaLR:
    warmup = max(1, cfg.warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, loader, criterion, device, vocab_size: int, amp: bool = False) -> dict:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model.forward_batch(batch)
            loss = criterion(logits.reshape(-1, vocab_size), batch["labels"].reshape(-1))
        n = batch["n_tgt_tokens"]
        total_loss += loss.item() * n
        total_tokens += n
    mean = total_loss / max(total_tokens, 1)
    return {"loss": mean, "ppl": math.exp(min(mean, 20))}


def train(model, tok, train_loader, valid_loader, device,
          cfg: TrainConfig = None, tag: str = "base", epochs: int = None) -> dict:
    cfg = cfg or CFG.train
    epochs = epochs or cfg.epochs
    ckpt_dir = Path(cfg.checkpoint_dir) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    criterion = build_loss(tok.pad_id, CFG.model.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.peak_lr, betas=cfg.betas,
                                  eps=cfg.eps, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = build_scheduler(optimizer, cfg, total_steps)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_val, bad_epochs = 0, float("inf"), 0
    history: list[dict] = []

    # ---- resume if a previous session was interrupted ----
    last = ckpt_dir / "last.pt"
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
        best_val = state["best_val"]
        history = state.get("history", [])
        print(f"resumed {tag} from epoch {start_epoch} (best val {best_val:.4f})")

    print(f"\ntraining '{tag}': {epochs} epochs x {len(train_loader):,} steps "
          f"= {total_steps:,} total | amp={use_amp} | device={device}")

    for epoch in range(start_epoch, epochs):
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)   # re-randomise buckets

        model.train()
        run_loss, run_tokens, t0 = 0.0, 0, time.time()

        for step, batch in enumerate(train_loader, 1):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model.forward_batch(batch)
                loss = criterion(logits.reshape(-1, tok.vocab_size),
                                 batch["labels"].reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                     # unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            n = batch["n_tgt_tokens"]
            run_loss += loss.item() * n
            run_tokens += n
            if step % cfg.log_every == 0:
                print(f"  epoch {epoch + 1} step {step:>5}/{len(train_loader)} "
                      f"loss {run_loss / run_tokens:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e}", flush=True)

        train_loss = run_loss / max(run_tokens, 1)
        val = evaluate(model, valid_loader, criterion, device, tok.vocab_size, use_amp)
        dt = time.time() - t0
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "val_loss": val["loss"], "val_ppl": val["ppl"], "seconds": dt})
        print(f"epoch {epoch + 1:>2}/{epochs}  train {train_loss:.4f}  "
              f"val {val['loss']:.4f}  ppl {val['ppl']:.2f}  ({dt:.0f}s)")

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch + 1, "best_val": best_val, "history": history,
                 "model_config": CFG.model.__dict__, "tag": tag}
        torch.save(state, ckpt_dir / "last.pt")

        if val["loss"] < best_val:
            best_val, bad_epochs = val["loss"], 0
            state["best_val"] = best_val
            torch.save(state, ckpt_dir / "best.pt")
            print(f"           new best -> {ckpt_dir / 'best.pt'}")
        else:
            bad_epochs += 1
            print(f"           no improvement ({bad_epochs}/{cfg.early_stop_patience})")
            if bad_epochs >= cfg.early_stop_patience:
                print(f"early stopping at epoch {epoch + 1}; "
                      f"validation bottomed at {best_val:.4f}")
                break

    (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
    return {"tag": tag, "best_val": best_val, "history": history,
            "best_ckpt": str(ckpt_dir / "best.pt")}
