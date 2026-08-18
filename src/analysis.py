"""
Milestone 4: quantitative analysis and error diagnostics.

Three things here that a plain BLEU number cannot tell you.

SIGNIFICANCE. A +1.3 BLEU gap between two systems means nothing on its own --
BLEU has sampling variance, and on a 2000-sentence test set a gap that size can
easily be noise. We use a paired bootstrap (Koehn, 2004): resample the test set
with replacement, recompute both systems' corpus BLEU on the same resample, and
count how often the gap reverses. Because corpus BLEU is not the mean of
sentence BLEUs, we cache per-sentence sufficient statistics (clipped n-gram
matches, n-gram totals, hypothesis and reference lengths) and re-aggregate them
per resample, which is both exact and fast.

LENGTH BUCKETS. Corpus BLEU is dominated by whatever length dominates the
corpus. On subtitle data that is short dialogue, so a headline BLEU of 25 can
hide near-total failure on long sentences. Bucketing exposes it.

ERROR CATEGORIES. Automatic detectors for truncation, degenerate repetition and
untranslated named entities. These are hints for a human to confirm, not
verdicts -- the report's error analysis should quote sentences the author has
actually read.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # project root on path

import math
import re
from collections import Counter

import numpy as np

from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a

_TOK13A = Tokenizer13a()
MAX_N = 4


# --------------------------------------------------------------------------
# BLEU via cached sufficient statistics
# --------------------------------------------------------------------------


def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def sentence_stats(hyp: str, ref: str) -> dict:
    """Per-sentence BLEU sufficient statistics under 13a tokenization."""
    h = _TOK13A(hyp).split()
    r = _TOK13A(ref).split()
    match, total = [0] * MAX_N, [0] * MAX_N
    for n in range(1, MAX_N + 1):
        hc, rc = _ngrams(h, n), _ngrams(r, n)
        total[n - 1] = max(0, len(h) - n + 1)
        match[n - 1] = sum(min(c, rc[g]) for g, c in hc.items())
    return {"hyp_len": len(h), "ref_len": len(r), "match": match, "total": total}


def corpus_bleu_from_stats(stats: list[dict]) -> float:
    """Aggregate cached statistics into corpus BLEU (sacreBLEU 'exp' smoothing)."""
    if not stats:
        return 0.0
    hyp_len = sum(s["hyp_len"] for s in stats)
    ref_len = sum(s["ref_len"] for s in stats)
    if hyp_len == 0:
        return 0.0

    log_sum, smooth = 0.0, 1.0
    for n in range(MAX_N):
        m = sum(s["match"][n] for s in stats)
        t = sum(s["total"][n] for s in stats)
        if t == 0:
            return 0.0
        if m == 0:
            smooth *= 2                       # exponential smoothing for zero counts
            p = 1.0 / (smooth * t)
        else:
            p = m / t
        log_sum += math.log(p)

    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    return 100.0 * bp * math.exp(log_sum / MAX_N)


def paired_bootstrap(stats_a: list[dict], stats_b: list[dict],
                     n_samples: int = 1000, seed: int = 1337) -> dict:
    """Is system B better than system A? Returns the observed gap and a p-value.

    Both systems are scored on the SAME resample each iteration, which is what
    makes the test paired and far more sensitive than comparing independent
    confidence intervals.
    """
    assert len(stats_a) == len(stats_b), "systems must cover the same sentences"
    n = len(stats_a)
    rng = np.random.default_rng(seed)

    base_a = corpus_bleu_from_stats(stats_a)
    base_b = corpus_bleu_from_stats(stats_b)
    observed = base_b - base_a

    wins, deltas = 0, []
    for _ in range(n_samples):
        idx = rng.integers(0, n, size=n)
        a = corpus_bleu_from_stats([stats_a[i] for i in idx])
        b = corpus_bleu_from_stats([stats_b[i] for i in idx])
        deltas.append(b - a)
        if b - a <= 0:
            wins += 1

    deltas = np.array(deltas)
    return {
        "bleu_a": round(base_a, 2),
        "bleu_b": round(base_b, 2),
        "delta": round(observed, 2),
        "p_value": round(wins / n_samples, 4),
        "ci95_low": round(float(np.percentile(deltas, 2.5)), 2),
        "ci95_high": round(float(np.percentile(deltas, 97.5)), 2),
        "n_samples": n_samples,
    }


# --------------------------------------------------------------------------
# Length-bucketed performance
# --------------------------------------------------------------------------


def bleu_by_length(hyps: list[str], refs: list[str], srcs: list[str],
                   edges=(0, 10, 20, 30, 50, 10_000)) -> list[dict]:
    """BLEU grouped by SOURCE length in whitespace tokens."""
    stats = [sentence_stats(h, r) for h, r in zip(hyps, refs)]
    lengths = np.array([len(s.split()) for s in srcs])

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.where((lengths >= lo) & (lengths < hi))[0]
        if len(idx) == 0:
            continue
        sub = [stats[i] for i in idx]
        hyp_len = sum(s["hyp_len"] for s in sub)
        ref_len = sum(s["ref_len"] for s in sub)
        rows.append({
            "bucket": f"{lo}-{hi - 1}" if hi < 10_000 else f"{lo}+",
            "n": int(len(idx)),
            "bleu": round(corpus_bleu_from_stats(sub), 2),
            "len_ratio": round(hyp_len / max(ref_len, 1), 3),
        })
    return rows


# --------------------------------------------------------------------------
# Error detectors
# --------------------------------------------------------------------------


_CAP = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
_FUNCTION_WORDS = {"The", "This", "That", "There", "They", "What", "When",
                   "Where", "Who", "Why", "How", "And", "But", "You", "Your",
                   "His", "Her", "Its", "Our", "Their", "It", "We", "He", "She",
                   "If", "For", "Not", "Are", "Was", "Were", "Have", "Has"}


def max_repeat(text: str, n: int = 3) -> int:
    """Largest number of times any n-gram repeats. >=3 signals degeneration."""
    toks = text.split()
    if len(toks) < n:
        return 0
    counts = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return max(counts.values()) if counts else 0


def untranslated_entities(src: str, hyp: str, ref: str) -> list[str]:
    """Capitalised source words copied verbatim into the hypothesis but absent
    from the reference -- i.e. the model failed to localise a name."""
    out = []
    for w in set(_CAP.findall(src)):
        if w in _FUNCTION_WORDS:
            continue
        if re.search(rf"\b{re.escape(w)}\b", hyp) and not re.search(
                rf"\b{re.escape(w)}\b", ref):
            out.append(w)
    return sorted(out)


def sentence_chrf(hyp: str, ref: str) -> float:
    import sacrebleu
    return sacrebleu.sentence_chrf(hyp, [ref]).score


def categorise(src: str, hyp: str, ref: str) -> list[str]:
    """Heuristic labels. Hints for a human reader, not verdicts."""
    tags = []
    h_len, r_len = len(hyp.split()), max(len(ref.split()), 1)
    ratio = h_len / r_len

    if ratio < 0.6:
        tags.append("truncation")
    elif ratio > 1.6:
        tags.append("over_generation")
    if max_repeat(hyp) >= 3:
        tags.append("repetition")
    if untranslated_entities(src, hyp, ref):
        tags.append("untranslated_entity")
    if not tags:
        tags.append("lexical_semantic")
    return tags


def find_failures(srcs, hyps, refs, top_k: int = 15) -> list[dict]:
    """Worst sentences by chrF, annotated with heuristic categories."""
    rows = []
    for i, (s, h, r) in enumerate(zip(srcs, hyps, refs)):
        if len(r.split()) < 3:            # 1-2 word refs give unstable scores
            continue
        rows.append({
            "index": i, "src": s, "hyp": h, "ref": r,
            "chrf": round(sentence_chrf(h, r), 2),
            "src_words": len(s.split()),
            "len_ratio": round(len(h.split()) / max(len(r.split()), 1), 2),
            "tags": categorise(s, h, r),
            "entities": untranslated_entities(s, h, r),
        })
    rows.sort(key=lambda x: x["chrf"])
    return rows[:top_k]


def category_summary(srcs, hyps, refs) -> dict:
    counts = Counter()
    for s, h, r in zip(srcs, hyps, refs):
        for t in categorise(s, h, r):
            counts[t] += 1
    n = max(len(srcs), 1)
    return {k: {"n": v, "pct": round(100 * v / n, 2)} for k, v in counts.most_common()}


# --------------------------------------------------------------------------
# Attention visualisation
# --------------------------------------------------------------------------


def cross_attention_matrix(model, tok, src_text: str, hyp_text: str, device):
    """Return (attn, src_pieces, tgt_pieces) for the final decoder layer,
    averaged over heads. Teacher-forces the produced hypothesis so the matrix
    lines up with the tokens the model actually emitted."""
    import torch

    model.eval()
    src_ids = tok.encode_source(src_text)
    tgt_ids = tok.encode_target(hyp_text)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    dec = torch.tensor([tgt_ids[:-1]], dtype=torch.long, device=device)

    with torch.no_grad():
        model(src, dec, src.eq(tok.pad_id), dec.eq(tok.pad_id), store_attn=True)

    attn = model.last_attn["cross"][-1][0].mean(0).float().cpu().numpy()  # (T, S)
    clean = lambda ids: [tok.sp.id_to_piece(int(i)).replace("\u2581", " ").strip()
                         or "_" for i in ids]
    return attn, clean(src_ids), clean(tgt_ids[:-1])


def plot_attention(attn, src_pieces, tgt_pieces, title: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(src_pieces) * 0.45),
                                    max(4, len(tgt_pieces) * 0.35)))
    im = ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(src_pieces)))
    ax.set_xticklabels(src_pieces, rotation=90, fontsize=8)
    ax.set_yticks(range(len(tgt_pieces)))
    ax.set_yticklabels(tgt_pieces, fontsize=8)
    ax.set_xlabel("source (English)")
    ax.set_ylabel("generated (Spanish)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
