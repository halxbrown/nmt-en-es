"""
Milestone 3, part 1: inference-time decoding.

Training uses teacher forcing -- the decoder always sees the correct prefix.
Inference has no such luxury: the model consumes its own predictions, so an
early mistake compounds. Greedy and beam search differ only in how much of that
risk they hedge.

GREEDY takes the highest-probability token at each step. Fast, and it commits
irreversibly: a token that looks best locally may leave no good continuation.

BEAM SEARCH keeps the k most probable partial sequences and lets them compete.
It approximates the highest-probability *sequence* rather than a chain of
locally best tokens. Roughly k times the compute for typically +1-2 BLEU.

LENGTH PENALTY. Sequence log-probability is a sum of negative terms, so longer
hypotheses always score worse and raw beam search systematically truncates. We
divide by the GNMT penalty ((5+len)/6)^alpha (Wu et al., 2016). alpha=0 disables
it; alpha near 1 approaches plain mean-log-probability.

No KV cache here: each step re-runs the decoder over the whole prefix, which is
O(T^2) overall. At these sequence lengths it is not the bottleneck, and the
simpler code is much easier to verify. Caching is listed under Future Work.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # project root on path

import numpy as np
import torch
import torch.nn.functional as F

NEG = -1e9  # finite stand-in for -inf: keeps topk and fp16 free of NaN


def _trim(ids, eos_id: int, pad_id: int) -> list[int]:
    """Drop BOS, cut at the first EOS, strip padding."""
    out = []
    for t in ids[1:]:
        t = int(t)
        if t == eos_id:
            break
        if t != pad_id:
            out.append(t)
    return out


@torch.no_grad()
def greedy_decode(model, src, src_key_padding_mask, bos_id, eos_id, pad_id,
                  max_len: int = 128):
    """Returns (B, L) token ids including BOS."""
    model.eval()
    device = src.device
    B = src.size(0)
    memory = model.encode(src, src_key_padding_mask)

    ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len):
        out = model.decode(ys, memory, ys.eq(pad_id), src_key_padding_mask)
        next_tok = model.generator(out[:, -1]).argmax(-1)
        # A finished sequence emits padding forever, so its row stops changing.
        next_tok = torch.where(finished, torch.full_like(next_tok, pad_id), next_tok)
        ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
        finished |= next_tok.eq(eos_id)
        if bool(finished.all()):
            break
    return ys


@torch.no_grad()
def beam_search(model, src, src_key_padding_mask, bos_id, eos_id, pad_id,
                beam_size: int = 5, max_len: int = 128, length_penalty: float = 0.6):
    """Batched beam search. Returns (B, L) ids of the best hypothesis per input."""
    model.eval()
    device = src.device
    B, K = src.size(0), beam_size

    memory = model.encode(src, src_key_padding_mask)          # (B, S, D)
    S, D = memory.size(1), memory.size(2)
    # Replicate each source K times so all beams decode in one batch.
    mem = memory.unsqueeze(1).expand(B, K, S, D).reshape(B * K, S, D)
    mem_pad = (src_key_padding_mask.unsqueeze(1).expand(B, K, S).reshape(B * K, S)
               if src_key_padding_mask is not None else None)

    ys = torch.full((B * K, 1), bos_id, dtype=torch.long, device=device)
    # Only beam 0 is live at step 0; otherwise all K beams expand the identical
    # BOS state and the top-k returns K copies of the same token.
    scores = torch.full((B, K), NEG, device=device)
    scores[:, 0] = 0.0
    scores = scores.view(-1)
    finished = torch.zeros(B * K, dtype=torch.bool, device=device)

    for _ in range(max_len):
        out = model.decode(ys, mem, ys.eq(pad_id), mem_pad)
        logp = F.log_softmax(model.generator(out[:, -1]).float(), dim=-1)   # (B*K, V)
        V = logp.size(-1)

        # A finished beam may only extend with <pad>, at zero cost, so its score
        # freezes and it still competes against live beams on equal terms.
        if bool(finished.any()):
            logp = logp.masked_fill(finished.unsqueeze(1), NEG)
            logp[finished, pad_id] = 0.0

        cand = (scores.unsqueeze(1) + logp).view(B, K * V)
        top_scores, top_idx = cand.topk(K, dim=-1)            # (B, K)
        beam_idx, tok_idx = top_idx // V, top_idx % V

        # Reorder the running hypotheses to follow the surviving beams.
        flat = (torch.arange(B, device=device).unsqueeze(1) * K + beam_idx).view(-1)
        ys = torch.cat([ys[flat], tok_idx.reshape(-1, 1)], dim=1)
        scores = top_scores.reshape(-1)
        finished = finished[flat] | tok_idx.reshape(-1).eq(eos_id)
        if bool(finished.all()):
            break

    # ---- length-normalise, then pick the best beam per source sentence ----
    seq = ys.view(B, K, -1)
    lengths = torch.full((B, K), seq.size(2) - 1, dtype=torch.float, device=device)
    for b in range(B):
        for k in range(K):
            row = seq[b, k, 1:]
            hit = (row == eos_id).nonzero()
            if len(hit):
                lengths[b, k] = float(hit[0].item() + 1)
    penalty = ((5.0 + lengths) / 6.0) ** length_penalty
    best = (scores.view(B, K) / penalty).argmax(dim=-1)
    return seq[torch.arange(B, device=device), best]


@torch.no_grad()
def translate_corpus(model, tok, sentences: list[str], device, method: str = "greedy",
                     batch_size: int = 32, max_len: int = 128, beam_size: int = 5,
                     length_penalty: float = 0.6, progress: bool = True) -> list[str]:
    """Translate raw strings, returning hypotheses in the ORIGINAL order.

    Sentences are sorted by length so each batch pads to a similar width, then
    the permutation is inverted. Forgetting to invert it silently misaligns
    hypotheses against references and produces a near-zero BLEU that looks like
    a model failure.
    """
    model.eval()
    encoded = [tok.encode_source(s)[:max_len] for s in sentences]
    order = np.argsort([len(e) for e in encoded])
    hyps: dict[int, str] = {}

    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        chunk = [encoded[i] for i in idx]
        width = max(len(c) for c in chunk)
        src = torch.full((len(chunk), width), tok.pad_id, dtype=torch.long, device=device)
        for r, c in enumerate(chunk):
            src[r, :len(c)] = torch.tensor(c, dtype=torch.long, device=device)
        pad_mask = src.eq(tok.pad_id)

        if method == "greedy":
            out = greedy_decode(model, src, pad_mask, tok.bos_id, tok.eos_id,
                                tok.pad_id, max_len)
        elif method == "beam":
            out = beam_search(model, src, pad_mask, tok.bos_id, tok.eos_id, tok.pad_id,
                              beam_size, max_len, length_penalty)
        else:
            raise ValueError(f"unknown method {method!r}")

        for r, i in enumerate(idx):
            hyps[int(i)] = tok.sp.decode(_trim(out[r].tolist(), tok.eos_id, tok.pad_id))

        if progress and (start // batch_size) % 10 == 0:
            print(f"    {method}: {min(start + batch_size, len(order))}/{len(order)}",
                  flush=True)

    return [hyps[i] for i in range(len(sentences))]
