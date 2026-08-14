"""
Milestone 2 driver: model definition, single-batch forward pass, masking
verification.

    python run_milestone2.py --smoke   # tiny synthetic corpus
    python run_milestone2.py           # real cached corpus from Milestone 1

Milestone 1's mask checks were structural -- they inspected the tensors. These
are behavioural: they run data through the network and check that information
cannot flow where it should not. That distinction matters, because a mask can
have a perfectly correct shape and still be applied to the wrong axis.
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

from config import ARTIFACT_DIR, CFG
from dataset import TranslationDataset, make_dataloader
from model import TransformerNMT, build_loss
from tokenizer import NMTTokenizer


def rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=60, help="overfit-test steps")
    args = ap.parse_args()

    torch.manual_seed(CFG.data.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prefix = "spm_smoke" if args.smoke else CFG.tok.model_prefix
    tok = NMTTokenizer(ARTIFACT_DIR / f"{prefix}.model", CFG.tok)
    ds = TranslationDataset.load(ARTIFACT_DIR / "train_encoded.npz")

    CFG.loader.num_workers = 0
    CFG.loader.pin_memory = False
    loader = make_dataloader(ds, tok, CFG.loader, train=True, bucket=True)
    batch = next(iter(loader))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    # ------------------------------------------------------------ 1. build
    rule("1. MODEL")
    model = TransformerNMT(tok.vocab_size, tok.pad_id, CFG.model).to(device)
    m = CFG.model
    print(f"device: {device}   vocab: {tok.vocab_size:,}")
    print(f"d_model={m.d_model} heads={m.n_heads} d_ff={m.d_ff} "
          f"layers={m.n_encoder_layers}+{m.n_decoder_layers} "
          f"dropout={m.dropout} pre_norm={m.pre_norm}\n")
    print(model.parameter_summary())

    tied = model.generator.weight.data_ptr() == model.embed.weight.data_ptr()
    print(f"\n  [{'PASS' if tied else 'FAIL'}] embedding / generator weights share storage "
          f"(3-way tying: {m.tie_embeddings})")

    # ------------------------------------------------------ 2. forward pass
    rule("2. SINGLE-BATCH FORWARD PASS")
    model.eval()
    with torch.no_grad():
        logits = model.forward_batch(batch, store_attn=True)

    B, T = batch["decoder_input"].shape
    print(f"  src              {tuple(batch['src'].shape)}")
    print(f"  decoder_input    {tuple(batch['decoder_input'].shape)}")
    print(f"  logits           {tuple(logits.shape)}")
    assert logits.shape == (B, T, tok.vocab_size), "logit shape mismatch"
    assert torch.isfinite(logits).all(), "non-finite values in logits"
    print(f"  [PASS] logits are (batch, tgt_len, vocab) and all finite")

    ea = model.last_attn["encoder_self"][0]
    ca = model.last_attn["cross"][0]
    print(f"  encoder self-attn {tuple(ea.shape)}  (B, heads, S, S)")
    print(f"  cross-attn        {tuple(ca.shape)}  (B, heads, T, S)")
    row_sums = ca.sum(-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)
    print(f"  [PASS] attention rows sum to 1")

    # ------------------------------------------------ 3. loss at initialisation
    rule("3. LOSS AT INITIALISATION")
    criterion = build_loss(tok.pad_id, CFG.model.label_smoothing)
    with torch.no_grad():
        loss0 = criterion(logits.reshape(-1, tok.vocab_size), batch["labels"].reshape(-1))
    lnV = torch.log(torch.tensor(float(tok.vocab_size))).item()

    # A tied + pre-norm model does NOT start uniform. The residual stream carries
    # embed(y_t) to the output, and tying makes generator row j identical to
    # embedding row j, so the untrained model's favourite prediction is the token
    # it was just fed. The label is the NEXT token, so loss starts above ln(V).
    # Quantify that bias rather than mistaking it for a bug.
    with torch.no_grad():
        keep = batch["labels"].ne(tok.pad_id)
        pick = lambda ids: logits.gather(-1, ids.unsqueeze(-1)).squeeze(-1)[keep].mean().item()
        mean_all = logits[keep].mean().item()
        mean_cur = pick(batch["decoder_input"])
        mean_tgt = pick(batch["labels"])

    print(f"  observed loss   {loss0.item():.4f}")
    print(f"  ln(vocab)       {lnV:.4f}   (loss of a perfectly uniform model)")
    print(f"  excess          {loss0.item() - lnV:+.4f}")
    print(f"\n  copy bias at init (explains the excess):")
    print(f"    mean logit, all tokens       {mean_all:+.3f}")
    print(f"    mean logit, input token y_t  {mean_cur:+.3f}  <- residual stream + tying")
    print(f"    mean logit, target y_(t+1)   {mean_tgt:+.3f}")

    # Below ln(V) at init is the dangerous direction: it means the model can
    # already see the answer, i.e. the causal mask leaks.
    if loss0.item() < lnV - 0.3:
        print(f"\n  [FAIL] loss BELOW ln(V) before training -- label leakage.")
    elif loss0.item() > lnV + 4.0:
        print(f"\n  [WARN] excess is large; check embedding init scale and sqrt(d_model).")
    else:
        print(f"\n  [PASS] loss sits just above ln(V), as expected for a tied pre-norm model.")

    # -------------------------------------------------- 4. causality (the big one)
    rule("4. CAUSAL MASK -- BEHAVIOURAL TEST")
    print("  Corrupt one target token, then check that predictions BEFORE it are")
    print("  unchanged. If the causal mask leaks, the model sees its own answer,")
    print("  training loss collapses, and BLEU at inference is near zero.\n")
    t = max(1, T // 2)
    corrupted = batch["decoder_input"].clone()
    corrupted[:, t] = (corrupted[:, t] + 7) % tok.vocab_size
    with torch.no_grad():
        logits_b = model(batch["src"], corrupted,
                         batch["src_key_padding_mask"], batch["tgt_key_padding_mask"])
    before = (logits[:, :t] - logits_b[:, :t]).abs().max().item()
    at_after = (logits[:, t:] - logits_b[:, t:]).abs().max().item()
    print(f"  max change at positions < {t}:  {before:.3e}   (must be ~0)")
    print(f"  max change at positions >= {t}: {at_after:.3e}   (must be non-zero)")
    assert before < 1e-4, "CAUSAL LEAK: the past changed when the future changed"
    assert at_after > 1e-4, "the perturbation had no effect at all -- test is not valid"
    print("  [PASS] information does not flow backwards in time")

    # ------------------------------------------- 5. padding invariance
    rule("5. PADDING MASK -- BEHAVIOURAL TEST")
    print("  Append extra <pad> columns to the source. Real-token predictions")
    print("  must not move. If they do, padding is contributing to attention.\n")
    pad_cols = 5
    B_, S_ = batch["src"].shape
    src_pad = torch.cat([batch["src"],
                         torch.full((B_, pad_cols), tok.pad_id,
                                    dtype=torch.long, device=device)], dim=1)
    with torch.no_grad():
        logits_c = model(src_pad, batch["decoder_input"],
                         src_pad.eq(tok.pad_id), batch["tgt_key_padding_mask"])
    delta = (logits - logits_c).abs().max().item()
    print(f"  max logit change after adding {pad_cols} pad columns: {delta:.3e}")
    assert delta < 1e-4, "padding is leaking into attention"
    print("  [PASS] output is invariant to source padding")

    # ------------------------------------------------- 6. gradient flow
    rule("6. BACKWARD PASS")
    model.train()
    logits_t = model.forward_batch(batch)
    loss = criterion(logits_t.reshape(-1, tok.vocab_size), batch["labels"].reshape(-1))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
    print(f"  parameters without gradient: {len(missing)}")
    print(f"  global gradient norm: {gnorm:.3f}")
    assert not missing, f"no gradient reached: {missing[:5]}"
    assert torch.isfinite(torch.tensor(gnorm)), "gradient norm is not finite"
    print("  [PASS] gradients reach every parameter and are finite")
    model.zero_grad(set_to_none=True)

    # ------------------------------------------- 7. overfit a single batch
    rule("7. OVERFIT ONE BATCH (capacity sanity check)")
    print("  A correct seq2seq model can memorise a single batch almost perfectly.")
    print("  If loss plateaus here, no amount of data or epochs will help.\n")
    small = {k: (v[:16] if torch.is_tensor(v) and v.dim() > 0 else v)
             for k, v in batch.items()}
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)
    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        out = model.forward_batch(small)
        l = criterion(out.reshape(-1, tok.vocab_size), small["labels"].reshape(-1))
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % max(1, args.steps // 5) == 0 or step == 1:
            print(f"    step {step:>4}   loss {l.item():.4f}")
    dt = time.time() - t0
    print(f"\n  {args.steps} steps in {dt:.1f}s  ({dt / args.steps * 1000:.0f} ms/step on {device})")
    final = l.item()
    verdict = "PASS" if final < loss0.item() * 0.6 else "WARN"
    print(f"  [{verdict}] loss fell from {loss0.item():.3f} to {final:.3f}")
    if verdict == "WARN":
        print("         Not yet memorised -- run more steps before concluding a bug.")

    rule("MILESTONE 2 COMPLETE")
    print("Model defined, forward pass clean, masking verified behaviourally.")
    print("Next: training loop with validation tracking, then beam search.")


if __name__ == "__main__":
    main()
