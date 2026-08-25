# A Pre-Norm Transformer for English→Spanish Translation: Capacity, Search, and Length Bias at 100k Sentence Pairs

**Rodrigo [Surname]** — Keiser University — Natural Language Processing

---

## Abstract

We implement a Transformer encoder–decoder for English→Spanish translation from
PyTorch primitives, trained on 100,000 sentence pairs sampled from OPUS-100. The
system uses a joint 16k SentencePiece BPE vocabulary with three-way weight
tying, pre-norm residual blocks, and manually implemented scaled dot-product
attention. We train two capacity presets (37.6M and 24.0M parameters) with
identical regularization and evaluate both with greedy and beam search on a
held-out test set, reporting paired-bootstrap significance for every comparison.
Beam search (k=5) improves BLEU significantly for both models (+1.06 and +0.88),
and the larger model outperforms the smaller by +1.01 BLEU (p < 0.001). We
identify a systematic length bias — the hypothesis/reference length ratio is
approximately 0.92 in *every* sentence-length bucket — and show that tuning the
GNMT length penalty on validation data recovers a further +0.36 and +0.53 BLEU.
The BLEU-optimal length ratio proves to be roughly 0.98 rather than parity:
driving hypotheses to reference length is achievable but reduces n-gram
precision more than it relieves the brevity penalty. We further document two
methodological failures found during development — a verification test that
passed vacuously, and a hyperparameter selection that transferred worse than a
smaller value despite a monotonically improving tuning curve.

---

## 1. Introduction and Motivation

### 1.1 Language pair and corpus

We translate English into Spanish using the `en-es` portion of OPUS-100
[Zhang et al., 2020], derived from OPUS [Tiedemann, 2012]. Spanish was chosen
as a *high-resource* pair despite the assignment's encouragement of low-resource
settings. The reasoning is methodological: with abundant clean data available,
any performance ceiling we observe is attributable to model capacity, training
budget, or decoding strategy rather than to data scarcity. This makes the
capacity ablation in Section 4 interpretable in a way it would not be if the
corpus were the binding constraint.

### 1.2 Corpus characteristics

The training split contains 1,000,000 sentence pairs, dominated by subtitle
dialogue — representative sentences include *"- Chris."* and *"Can we dim the
house lights a little bit more?"*. Three consequences shape the entire
evaluation.

The register is informal and conversational, not the formal register of Europarl
or news. Sentences are short: the median source is 12 subword tokens against a
mean of 18.3, a strongly right-skewed distribution. And subtitle lines are
*context-dependent fragments* — a reference may legitimately contain material
the source does not, because the human subtitler had surrounding dialogue. We
return to this in Section 5.

### 1.3 Goals

Our objectives are (i) to implement the Transformer from primitive components,
(ii) to verify correctness behaviourally rather than by inspection, (iii) to
quantify the contributions of capacity and decoding search *separately* with
significance testing, and (iv) to characterize failure modes concretely enough
to motivate specific improvements.

---

## 2. System Architecture

### 2.1 Data pipeline and filtering

Pairs pass through NFKC normalization followed by eight filters (Table 1).

**Table 1: Training-data filtering (1,000,000 input pairs)**

| Filter | Dropped | % |
|---|---:|---:|
| Duplicate source (same EN, different ES) | 84,064 | 8.41 |
| Duplicate pair (exact) | 35,257 | 3.53 |
| Length ratio > 2.0 | 15,755 | 1.58 |
| Identical source and target | 15,194 | 1.52 |
| Non-alphabetic (< 50% letters) | 7,977 | 0.80 |
| Source length outside [1, 80] words | 2,726 | 0.27 |
| Target length outside [1, 80] words | 2,083 | 0.21 |
| Contains URL | 1,587 | 0.16 |
| **Total dropped** | **164,643** | **16.46** |
| **Retained** | **835,357** | **83.54** |

Two entries deserve comment. The 8.41% duplicate-source rate reflects OPUS's
construction from overlapping subtitle releases; retaining multiple targets for
one source would give the model contradictory supervision for identical input.
The 1.52% flagged identical is untranslated English in the Spanish column — a
known OPUS artifact, and copy-noise that teaches the model the wrong behaviour.

From the 835,357 retained pairs we sample 100,000 with seed 1337. We use the
*official* OPUS-100 validation and test splits (2,000 each), which the corpus
authors guarantee not to overlap training data — a guarantee a random split
could not provide given the duplication rate above.

Aggressive filtering is applied **only** to training. Validation and test
receive normalization and empty-line removal alone; filtering the test set would
make it easier than the true input distribution and inflate BLEU.

### 2.2 Tokenization

We train a joint SentencePiece [Kudo and Richardson, 2018] BPE model
[Sennrich et al., 2016] of 16,000 pieces over both language sides. A joint
vocabulary suits this pair because English and Spanish share the Latin alphabet
and a large cognate inventory (*information*/*información*); shared subwords
occupy one embedding row, numbers and names tokenize identically on both sides,
and the vocabulary can be tied three ways (Section 2.3). The size is matched to
the corpus: 32k pieces over 3.79M subword tokens would leave the rarest rows
undertrained.

Configuration: `pad_id=0` (so a zero-initialized tensor is naturally
all-padding), `split_digits=True`, `byte_fallback=True` (unknown characters
decompose to UTF-8 byte tokens rather than `<unk>`), and `nmt_nfkc`
normalization. The tokenizer is trained on the **training split only**; fitting
it on validation or test text is a mild but real leak.

**Table 2: Tokenizer health**

| Split | Fertility | UNK rate |
|---|---:|---:|
| train.en | 1.49 | 0.0000% |
| train.es | 1.49 | 0.0000% |
| valid.en (unseen) | 1.53 | 0.0000% |
| valid.es (unseen) | 1.52 | 0.0000% |

Identical fertility across languages indicates the joint vocabulary is not
skewed toward either side. The drift to 1.53/1.52 on unseen text is small. The
zero UNK rate follows from `byte_fallback`, which matters for the error analysis
because it distinguishes *"rare token translated badly"* from *"token was never
representable."*

### 2.3 Model

Both presets share a pre-norm encoder–decoder with manually implemented scaled
dot-product attention:

```
Attention(Q,K,V) = softmax(QKᵀ / √d_head + M) V
```

`M` is an additive mask filled with `finfo.min` at forbidden positions — not
`-inf`, because a fully masked softmax row over `-inf` produces NaN and `-inf`
overflows under fp16 autocast.

We implement attention manually rather than calling `nn.MultiheadAttention`
for two reasons: every module returns its attention weight matrix, so the
visualizer in Section 5.4 needs no forward hooks; and the mask application is
legible where a reader can verify it.

**Pre-norm residual blocks.** The original Transformer [Vaswani et al., 2017]
normalizes *after* the residual addition, producing large gradients at the top
of the stack early in training and effectively requiring the inverse-square-root
warmup schedule to converge [Xiong et al., 2020]. We place LayerNorm inside the
residual branch, which is far less sensitive to learning rate — valuable when
the compute budget does not permit many failed runs. A final LayerNorm follows
each stack; without it the output scale grows with depth and the softmax
saturates.

**Three-way weight tying.** Source embedding, target embedding, and output
projection share a single 16000 × 512 matrix [Press and Wolf, 2017]. This is
legitimate only because the vocabulary is joint: token ID 4021 denotes the same
string in both languages. Tying removes ~8.2M parameters and regularizes.

**Positional encoding.** Fixed sinusoidal, chosen over learned positions because
the model must handle test sentences longer than any seen in 100k pairs;
sinusoids extrapolate, learned embeddings do not. Embeddings are initialized
with standard deviation `d_model^-0.5` and scaled by `√d_model` so the sinusoids
do not swamp token identity.

**Table 3: Capacity presets**

| | base | small |
|---|---:|---:|
| Encoder / decoder layers | 4 + 4 | 3 + 3 |
| `d_model` | 512 | 512 |
| Attention heads | 8 | 8 |
| `d_ff` | 2048 | 1024 |
| Dropout | 0.2 | 0.2 |
| Parameters (tied) | 37,619,712 | 23,965,696 |

Both are deliberately shallower than the paper's 6+6 base (~65M). With 3.79M
training subwords, 65M parameters is roughly seventeen per token of signal;
Sennrich and Zhang [2019] show lower-resource NMT benefits from reduced depth
and heavier regularization. **Dropout is held at 0.2 for both**, so the ablation
isolates capacity rather than confounding it with regularization strength.

### 2.4 Input pipeline

Sequences are tokenized once and stored as a flat `int32` array plus offsets
rather than a Python list of lists — this avoids repeated SentencePiece calls
and the refcount-touching that defeats `fork`'s copy-on-write in DataLoader
workers.

Batches pad dynamically to the longest sequence *in the batch*. Given a median
source length of 12 against a 128-token ceiling, static padding would spend the
overwhelming majority of attention FLOPs on `<pad>`.

Dynamic padding only helps if similar-length sequences co-occur, so a custom
sampler shuffles indices, cuts megabatches of `batch_size × 50`, sorts each by
the *sum* of source and target length, cuts batches, and shuffles batch order.
Sorting on source alone leaves target-side padding unaddressed.

**Table 4: Padding efficiency (100 batches)**

| Strategy | Useful cells | Wasted |
|---|---:|---:|
| Random batching | 22.8% | 77.2% |
| Length-bucketed (src+tgt) | 78.0% | 22.0% |

A **3.42× reduction in wasted attention cells per epoch**. Sorting on combined
rather than source-only length accounted for 3.4 points of this (74.6% → 78.0%).

---

## 3. Experimental Setup

### 3.1 Hyperparameters

**Table 5: Training configuration**

| Parameter | Value |
|---|---|
| Optimizer | AdamW, β=(0.9, 0.98), ε=1e-9 |
| Weight decay | 0.01 |
| Peak learning rate | 5e-4 |
| Schedule | 2,000-step linear warmup → cosine to 2% of peak |
| Batch size | 64 sentences (1,563 steps/epoch) |
| Loss | Cross-entropy, label smoothing 0.1, `ignore_index=pad` |
| Gradient clipping | 1.0 global norm, after unscaling |
| Precision | fp16 autocast with GradScaler |
| Epochs | 20, early stopping patience 3 on validation loss |
| Max sequence length | 128 subwords per side |
| Seed | 1337 |
| Hardware | NVIDIA Tesla T4 (Google Colab) |

Warmup serves a narrower purpose than in the original paper: with pre-norm
blocks it is not required for stability, only to prevent Adam's second-moment
estimates being fixed by a few noisy early batches. 2,000 steps ≈ 1.3 epochs.

Loss averages over non-padding positions via `ignore_index`, keeping gradient
scale comparable across batches whose token counts differ several-fold under
sentence-count batching.

Checkpoints write to Google Drive every epoch carrying optimizer, scheduler and
GradScaler state, so an interrupted session resumes rather than restarting with
a cold optimizer.

### 3.2 Correctness verification

A masking error degrades quality *without raising an exception*, so we verify
behaviourally. Four structural assertions (padding masks match pad positions;
padding is right-aligned; decoder input and labels offset by exactly one; the
causal mask blocks the strict upper triangle) are supplemented by five
behavioural tests.

**Causality.** Corrupting the target token at position *t* must leave every
prediction at positions < *t* bit-identical. Observed: maximum change
`0.000e+00` before *t*, `1.040e+01` at and after.

**Padding invariance.** Appending five `<pad>` columns to the source must leave
all logits unchanged. Observed `0.000e+00` on CPU and `5.722e-06` on GPU — cuBLAS
selecting different reduction orders for different tensor widths, since
floating-point addition is not associative. Two orders of magnitude inside the
1e-4 tolerance and six below the scale of a real leak.

**Decoder equivalence.** Beam search with k=1 and α=0 must reproduce greedy
token-for-token, since with one beam there is nothing to compare and the
algorithm degenerates to argmax. Observed 8/8 for both models.

**Search effectiveness.** Beam search with k=5 must find sequences of higher
model log-probability than greedy. On a length-diverse probe set (source lengths
9–128 subwords), base improved mean sequence log-probability from −15.005 to
−11.230, strictly better on 4/8 sentences and never worse; small improved from
−25.003 to −11.485, strictly better on 2/8 and never worse.

That beam is *strictly* better on only half the sentences is expected rather
than weak: on the remainder greedy already found the highest-probability
sequence and beam correctly returned it. Beam search improves the subset where
the greedy path is locally attractive but globally suboptimal, which is
consistent with a corpus-level gain of ~1 BLEU concentrated in a minority of
sentences.

We also note that small shows a *larger* log-probability improvement than base
while gaining *less* BLEU. The weaker model's greedy path lies further from
optimal, so search has more to recover in model-score terms — but recovering
model probability is not the same as recovering translation quality. Both models
converge to nearly the same beam log-probability (−11.23 and −11.49) from very
different greedy starting points.

**Length penalty.** Mean hypothesis length must be non-decreasing in α.
Observed 20.9 → 21.5 → 22.8 subwords for base and 21.5 → 21.5 → 27.2 for small
across α ∈ {0, 1, 2.5}.

**A verification test that passed vacuously.** The first version of the decoder
test drew its probe sentences from the first batch of a length-bucketed loader.
Because the sampler sorts by length, this returned the *eight shortest*
validation sentences — mean hypothesis length 3.2 subwords. On inputs that
short, greedy and beam produce identical output and no length penalty can
lengthen anything, so the search-effectiveness and length-penalty tests reported
PASS while testing nothing. All three checks passed; two were meaningless.

We record this because it is a general hazard: a test that passes for the wrong
reason is worse than no test, since it manufactures confidence. The fix was to
sample across the length distribution, and the numbers above come from the
corrected version.

**Reproducibility.** The pipeline produces byte-identical statistics on
Windows/CPU (Python 3.11) and Linux/T4 (Python 3.12) — identical filter counts,
fertility, padding efficiency, and the same sentence drawn into the same batch
position. All randomness derives from `seed=1337`.

### 3.3 Interpreting the initial loss

At initialization loss is 10.2690 against `ln(16000) = 9.6803`, an excess of
+0.5886 — expected, not anomalous. Weight tying makes output-projection row *j*
identical to embedding row *j*, and the pre-norm residual stream carries
`embed(y_t)` to the output largely unchanged, so the untrained model's strongest
prediction is *the token it was just fed*. Measured: mean logit of the input
token is +8.032 against −0.005 for all tokens. Since the label is the *next*
token, loss necessarily begins above uniform.

The diagnostic value lies in the opposite direction: loss *below* `ln(V)` before
training would indicate the model can already see its answer — a causal leak.

---

## 4. Results and Evaluation

All numbers are computed on the 2,000-sentence held-out test set, used neither
for training nor checkpoint selection. We report sacreBLEU [Post, 2018] with
signature `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`; BLEU is
not comparable across tokenizations, so the bare number is not reproducible
without it. We also report chrF [Popović, 2015] and ROUGE-L [Lin, 2004].

### 4.1 Training behaviour

**Table 6: Training summary**

| | base | small |
|---|---:|---:|
| Epochs completed | 20 / 20 | 20 / 20 |
| Best validation loss | 3.4754 (epoch 19) | 3.5706 (epoch 19) |
| Validation perplexity | 32.31 | 35.54 |
| Seconds / epoch | 129–144 | 101–110 |

Neither model triggered early stopping. Validation loss improved monotonically
through epoch 19, rising by only 0.0026 (base) at epoch 20. **Neither model
overfit**, despite base carrying roughly ten parameters per training subword;
dropout 0.2, label smoothing and weight tying were sufficient.

An important caveat: the learning rate reached its cosine floor (1.0e-5)
precisely as the curves flattened. We therefore *cannot* distinguish convergence
from the schedule ending. The evidence supports the claim that the binding
constraint was training budget rather than capacity, but not the stronger claim
that either model converged.

### 4.2 Decoding strategy

**Table 7: Decoding on the test set**

| Model | Decoding | BLEU | chrF | ROUGE-L | Len. ratio | sent/s |
|---|---|---:|---:|---:|---:|---:|
| base | greedy | 24.82 | 49.67 | 49.64 | — | 91.1 |
| base | beam k=5, α=0.6 | 25.88 | 50.49 | 50.84 | 0.918 | 18.0 |
| base | beam k=5, α=4.0 (tuned) | **26.24** | — | — | 0.991 | 18.0 |
| small | greedy | 23.99 | 48.80 | 49.00 | — | 136.4 |
| small | beam k=5, α=0.6 | 24.87 | 49.70 | 50.28 | 0.911 | 28.8 |
| small | beam k=5, α=2.2 (tuned) | 25.40 | — | — | 0.974 | 28.8 |

**Table 8: Paired-bootstrap significance (1,000 resamples)**

| Comparison | Δ BLEU | 95% CI | p |
|---|---:|---|---:|
| base: beam − greedy | +1.06 | [+0.69, +1.46] | < 0.001 |
| small: beam − greedy | +0.88 | [+0.54, +1.23] | < 0.001 |
| beam: base − small | +1.01 | [+0.50, +1.49] | < 0.001 |

A BLEU difference is meaningless without a variance estimate, so every
comparison uses a paired bootstrap [Koehn, 2004]: the test set is resampled with
replacement, both systems rescored on the *same* resample, and we count how
often the difference reverses. Because corpus BLEU is not the mean of sentence
BLEUs, per-sentence sufficient statistics (clipped n-gram matches, totals,
lengths) are cached once and re-aggregated per resample. Our aggregation was
validated against sacreBLEU to six decimal places across four regimes spanning
1.57 to 67.76 BLEU.

All three comparisons are significant, and the beam improvement appears in chrF
and ROUGE-L as well — it is not a BLEU artifact.

The beam gain is numerically larger for base (+1.06) than small (+0.88), but the
confidence intervals overlap heavily; this difference is **not established**. On
the validation set the two gains were +1.27 and +1.26, whose near-equality was
coincidence rather than evidence of independence between capacity and search.

Validation-to-test drops were modest (base beam 26.32 → 25.88; greedy 25.05 →
24.82), quantifying the optimism introduced by selecting checkpoints on
validation.

### 4.3 The capacity/search trade-off

Comparing *across* models rather than within them exposes a trade-off — and one
that changes once decoding is tuned.

Under the default α=0.6, base with greedy (24.82) and small with beam (24.87)
were statistically indistinguishable, suggesting parameters and search width
were interchangeable at this scale. Tuning the length penalty overturns this.
Greedy decoding has no length-penalty parameter, so only the beam configurations
improve: small with tuned beam reaches 25.40 and now **exceeds** base with
greedy by 0.58 BLEU.

The honest characterization is a latency/quality trade rather than an
equivalence. base+greedy decodes at 91.1 sentences/second against 28.8 for
small+tuned-beam — 3.2× faster for 0.58 BLEU less. Which is preferable depends
on whether the deployment constraint is throughput or quality. (We did not run a
paired bootstrap on this cross-model comparison, so 0.58 should be read as
indicative rather than established.)

At matched tuned decoding, capacity retains a clear advantage: base 26.24
against small 25.40, a gap of +0.84 broadly consistent with the +1.01 measured
at α=0.6. The cost is +13.65M parameters (+57%) and ~1.6× slower decoding.

The methodological lesson generalizes: had we compared decoding strategies only
at a single default hyperparameter, we would have reported a conclusion that
reverses under tuning. Ablations must hold *tuned* configurations against each
other.

### 4.4 Performance by sentence length

**Table 9: BLEU by source length, beam k=5, α=0.6**

| Source words | n | base BLEU | base ratio | small BLEU | small ratio |
|---|---:|---:|---:|---:|---:|
| 0–9 | 1,103 | 26.29 | 0.926 | 25.19 | 0.921 |
| 10–19 | 518 | 27.48 | 0.933 | 27.28 | 0.920 |
| 20–29 | 196 | 25.81 | 0.926 | 24.22 | 0.925 |
| 30–49 | 146 | 25.79 | 0.906 | 24.62 | 0.912 |
| 50+ | 37 | 21.03 | 0.866 | 19.43 | 0.825 |

Two observations, one anticipated and one not.

**Long-sequence degradation is gradual, not catastrophic.** We expected sharp
collapse beyond the training length distribution; instead BLEU declines slowly
and only the 50+ bucket drops materially. With **n = 37**, that bucket's 21.03
carries wide error bars and should be treated as suggestive. Sinusoidal
positional encodings and a 128-token ceiling that truncated only 93 of 200,000
sequences (0.046%) both plausibly contribute.

**Very short sentences underperform medium ones.** The 0–9 bucket holds 55% of
the test set yet scores below 10–19 for both models. Two mechanisms are
plausible: short subtitle lines are context-dependent fragments whose correct
translation is underdetermined by the source alone, and BLEU is unstable on
short sequences where a single wrong token eliminates a large fraction of the
available n-grams.

### 4.5 A systematic length bias

The ratio column of Table 9 reveals what headline BLEU conceals: the
hypothesis/reference length ratio is approximately **0.92 in every bucket**, not
merely the long ones. The model under-generates by roughly 8% uniformly.

Since sequence log-probability is a sum of negative terms, beam search
intrinsically prefers shorter hypotheses, and the GNMT length penalty
[Wu et al., 2016] at the default α=0.6 evidently under-corrects. We therefore
swept α on the **validation** set and applied only the winning value to test.
Tuning a decoding hyperparameter by test BLEU would contaminate the test set
exactly as selecting a checkpoint on test would; the reported figure comes from
a single test decode, not the best of nine.

**Table 10: Length-penalty sweep (1,000 validation sentences, beam k=5)**

| α | base BLEU | base ratio | small BLEU | small ratio |
|---:|---:|---:|---:|---:|
| 0.6 | 25.85 | 0.892 | 24.64 | 0.882 |
| 1.0 | 26.16 | 0.909 | 24.91 | 0.897 |
| 1.4 | 26.26 | 0.926 | 25.23 | 0.918 |
| 1.8 | 26.32 | 0.941 | 25.33 | 0.932 |
| 2.2 | 26.37 | 0.949 | **25.34** | 0.946 |
| 2.6 | 26.57 | 0.957 | — | — |
| 3.0 | 26.59 | 0.962 | — | — |
| 3.5 | 26.69 | 0.967 | — | — |
| 4.0 | **26.70** | 0.972 | — | — |

**Table 11: Tuned length penalty on test (1,000 resamples)**

| Model | α=0.6 | tuned | Δ BLEU | 95% CI | p |
|---|---:|---:|---:|---|---:|
| base (α=4.0) | 25.88 | 26.24 | +0.36 | [+0.02, +0.63] | 0.021 |
| small (α=2.2) | 24.87 | 25.40 | +0.53 | [+0.29, +0.78] | < 0.001 |

Three observations follow.

**The recoverable gain is far smaller than the brevity penalty implies.** A naïve
calculation divides 25.88 by the brevity penalty at ratio 0.918
(`exp(1 − 1/0.918) ≈ 0.915`) and predicts an unpenalized score near 28. That
reasoning holds n-gram precision constant while increasing length, which is
false: the additional tokens emitted under a large penalty are frequently wrong,
so precision falls as the penalty is relieved and the effects substantially
cancel. The ~2 BLEU implied by the brevity penalty is an upper bound on a
quantity decoding alone cannot extract.

**The BLEU-optimal length ratio is approximately 0.98, not 1.0.** The sweep
crosses this optimum: at α=2.6 the test ratio is 0.979 for 26.33 BLEU, while at
α=4.0 it reaches 0.991 for 26.24. Driving hypotheses to reference length is
achievable but costs more precision than it recovers in brevity penalty. Systems
should not be tuned toward a length ratio of 1.0 by default.

**Validation selection on a flat curve is noise-limited.** base's validation
BLEU rises monotonically across the entire swept range (25.85 → 26.70), but the
spread above α ≈ 1.8 is only 0.38 BLEU on 1,000 sentences — within sampling
variation. Selecting the validation argmax (α=4.0) transferred *worse* to test
(26.24) than α=2.6, the argmax of a narrower initial sweep, did (26.33). The
widened confidence interval reflects this directly: [+0.02, +0.63] at α=4.0
against [+0.23, +0.69] at α=2.6, with p rising from below 0.001 to 0.021.

We report the α=4.0 figure because it is what the stated protocol selects.
Reporting 26.33 instead would mean choosing α by test performance — the
contamination the validation sweep exists to prevent, and the fact that it is
the more flattering number is precisely why it must not be selected. The honest
characterization is that the tunable gain lies in the +0.36 to +0.45 range and
our procedure cannot resolve it more finely without a larger validation set.

---

## 5. Error Analysis and Discussion

**Table 12: Heuristic error categories, beam k=5, α=0.6**

| Category | base n | base % | small n | small % |
|---|---:|---:|---:|---:|
| Lexical / semantic (no structural flag) | 1,605 | 80.25 | 1,593 | 79.65 |
| Untranslated named entity | 218 | 10.90 | 228 | 11.40 |
| Truncation (ratio < 0.6) | 122 | 6.10 | 124 | 6.20 |
| Over-generation (ratio > 1.6) | 82 | 4.10 | 80 | 4.00 |
| Degenerate repetition | 6 | 0.30 | 9 | 0.45 |

Near-identical rates across presets indicate these failure modes are properties
of the corpus and tokenizer rather than of capacity. The detectors are heuristic
and produce false positives — a name flagged as untranslated may simply have
been paraphrased in the reference — so the examples below were read
individually.

### 5.1 Failure 1: idiomatic multi-word expressions

> **SRC** Here are the sites of each of those that have taken place:
> **REF** Estos son los sitios en que cada Congreso ha tenido lugar:
> **base** Aquí están los sitios de cada uno de esos que **se han llevado a cabo**:
> **small** Aquí están los sitios de cada uno de esos que **han tomado lugar**:

The clearest capacity effect we observed. base renders *"have taken place"* with
the correct Spanish idiom *llevarse a cabo*; small produces *tomar lugar*, a
word-for-word calque that is not Spanish. Idiomatic expressions require the
model to suppress the compositional reading in favour of a memorized mapping,
and the additional layers and wider feed-forward evidently support this.

Note also that the reference contains *"cada Congreso"* — a congress — appearing
nowhere in the source. This is the subtitle-fragment problem from Section 1.2:
the human subtitler had context the model does not. Both systems are arguably
*more* faithful to the given source than the reference is, and BLEU penalizes
them regardless. Sentence-level evaluation on subtitle corpora systematically
understates quality for this reason.

### 5.2 Failure 2: named entities and the personal *a*

> **SRC** I'm the man who killed Blackbeard.
> **REF** Sí. Soy el hombre que mató **a Barbanegra**.
> **base** Yo soy el hombre que mató **Blackbeard**.
> **small** Soy el hombre que mató **a Blackbeard**.

Two independent failures. Neither model localizes *Blackbeard* to *Barbanegra*.
Given `byte_fallback` this is not an OOV failure — the token was representable —
but a failure to learn a low-frequency entity mapping from 100k pairs. At 10.9%
of the test set, this is the largest identifiable non-generic category.

More interesting is that **small produces the required personal *a* and base
omits it**. Spanish marks human direct objects with *a*; *mató Blackbeard* is
ungrammatical. The larger model, which wins on aggregate BLEU by a statistically
significant margin, fails a specific grammatical constraint the smaller model
satisfies. This is direct evidence that corpus-level BLEU averages over
per-phenomenon behaviour and can invert it. A targeted contrastive evaluation
set would be required to measure such constraints properly.

The reference's leading *"Sí."* is again bleed from an adjacent subtitle line.

### 5.3 Failure 3: syntactic recovery under beam search

> **SRC** I don't even remember what the fight was about.
> **REF** No recuerdo por qué fue la pelea.
> **base greedy** Ni siquiera recuerdo **lo que la pelea estaba**.
> **base beam** Ni siquiera recuerdo **lo de la pelea**.
> **small greedy** Ni siquiera recuerdo lo que **se había sobre eso**.
> **small beam** Ni siquiera recuerdo lo que la pelea estaba.

This illustrates the mechanism behind Table 8. base's greedy output calques the
English interrogative complement, stranding the copula *estaba* at the end —
locally plausible tokens leaving no grammatical continuation. Beam search
escapes to *"lo de la pelea"*, grammatical though it loses the aboutness
relation. Note also the quality ladder: small's beam output equals base's
*greedy* output, consistent with Section 4.3.

### 5.4 Attention behaviour

Cross-attention maps from the final decoder layer (averaged over heads) show a
clear diagonal band for short sentences, indicating learned approximate
monotonic word alignment. No alignment supervision appears in the objective;
this is an emergent consequence of the encoder–decoder attention bottleneck.
Off-diagonal mass concentrates where English and Spanish word order diverges,
notably around adjective–noun inversion.

### 5.5 Out-of-distribution probing

The examples above come from the test set, which shares the training
distribution. We additionally probed the deployed translator with
self-authored inputs. Two patterns emerged that the test-set analysis does not
surface.

**In-domain conversational English works.** *"where is my head"* →
*"¿Dónde está mi cabeza?"*; *"oh my goodness, who's that friend?"* → *"Dios mío,
¿quién es ese amigo?"*, correctly rendering the interjection idiomatically
rather than literally. Both are subtitle-register dialogue.

**Rare content words fail through subword composition, not retrieval.** The
clearest case:

| Input | Output |
|---|---|
| `peanuts` | **Cacahuetes** ✓ |
| `i really enjoy eating peanuts` | *"Estoy disfrutando de comer* **pacahuetes***"* |
| `i enjoy eating peanuts` | *"Me gusta comer* **pacahujos***"* |
| `i truly enjoy eating peanuts` | *"Me gusta comer* **pastillas de comanuts***"* |
| `i really enjoy eating lightly salted peanuts` | *"Me gusta comer las* **manchas lágrimas de luz***"* |
| `hello, i'm really enjoying these dry roasted peanuts` | *"...estas* **manchas secas secas***"* |

The model produces *cacahuetes* correctly in isolation but cannot retrieve it in
context. Critically, the failures are not random substitutions: **pacahuetes**
is *cacahuetes* with the first two syllables transposed, and **pacahujos** is a
near-miss of the same word. This is characteristic of a rare word being
*composed* subword by subword rather than retrieved as a unit — each piece is
locally high-probability given the prefix, but the assembled string is not a
real word. Longer prefixes give the decoder more opportunity to drift, which is
consistent with the isolated single-word input succeeding.

In the longest inputs the noun is lost entirely, replaced by *manchas* (stains)
in both adjective-bearing cases — an attractor for that slot when the noun
cannot be resolved. *"lightly salted"* → *"lágrimas de luz"* is a separate
morphological failure: *lightly* was read as *light* (illumination) rather than
as a manner adverb, since the `-ly` suffix carrying the grammatical role is
split from the stem by subword segmentation.

**Case and punctuation shift lexical choice.** *"Liar liar pants on fire"* →
*"Piar los pantalones de mentiras en el fuego"*, where *Piar* is a real Spanish
verb (to chirp); *"liar, liar, pants on fire"* → *"mentiroso, mentirosa,
pantalones de fuego"*, correctly identifying the noun. The two inputs differ in
both capitalization and punctuation, so this comparison does not isolate either
variable — but it demonstrates that surface form materially changes lexical
selection, consistent with the `case:mixed` sacreBLEU setting.

These probes support the domain argument of Section 1.2 with evidence
independent of the test set: what works is subtitle dialogue, and what fails is
food and product description, a register OpenSubtitles barely contains.

### 5.6 Threats to validity

Single reference per source, so legitimate paraphrases are penalized. The
subtitle domain limits generalization to formal registers. Length-bucket
estimates for 50+ words rest on 37 sentences. The heuristic error categories are
unvalidated against human judgement. The out-of-distribution probes in Section
5.5 were author-selected and are illustrative, not a sample. Finally, both
models were trained once with a single seed; the +1.01 capacity gap is
significant under resampling of the *test set*, but we did not estimate variance
across training runs, which is typically comparable in magnitude.

---

## 6. Conclusion and Future Work

We built a pre-norm Transformer from PyTorch primitives, trained it on 100k
English–Spanish pairs, and reached 26.24 BLEU on a held-out test set. Every
architectural claim was verified behaviourally and every numerical comparison
carries a paired-bootstrap confidence interval.

Beam search yields a significant ~+1 BLEU for both capacity presets, and tuning
the length penalty adds a further +0.36 to +0.53 — roughly +1.4 BLEU available
at inference time with no retraining. Capacity retains a +0.84 advantage at
matched tuned decoding. The model under-generates by a near-constant ~8% at
every sentence length, and the BLEU-optimal ratio is ~0.98 rather than parity.

Two methodological findings are, we think, more transferable than the scores.

First, comparing decoding strategies at a single default hyperparameter produced
a conclusion — parity between base+greedy and small+beam — that reversed once
both configurations were tuned. Component ablations must compare tuned settings.

Second, the length-penalty selection failed instructively. Validation BLEU rose
monotonically across the entire swept range, but the spread above α ≈ 1.8 was
smaller than the sampling noise on a 1,000-sentence validation set, and the
validation argmax transferred worse to test than a smaller value. A
monotonically improving tuning curve is not evidence that the largest value is
best; without a variance estimate on the tuning set, the argmax of a flat curve
is close to an arbitrary choice.

We would add a third, from Section 3.2: our first decoder-verification test
passed while testing nothing, because its probe sentences were drawn from the
shortest end of the length distribution. Tests that pass for the wrong reason
are worse than absent tests.

**Deployment.** Training and inference share the same model and decoding
modules, so no separate deployment implementation can drift from the evaluated
one; the reported BLEU describes exactly the code path that runs at inference.
The training checkpoint (430.7 MB, carrying AdamW moments and scheduler state)
exports to a 143.6 MB inference-only checkpoint, a 67% reduction.

**Future work.** *A larger tuning set*: length-penalty selection is noise-limited
at 1,000 validation sentences; tuning on the full 2,000, or averaging across
bootstrap resamples rather than taking a point argmax, would make selection
reproducible. *Extended training*: neither model converged before the cosine
schedule expired, so a longer schedule would establish whether 26.24 reflects
the architecture or the budget. *Incremental decoding*: beam search re-runs the
decoder over the full prefix each step, giving O(T²) work; a key–value cache
would make it substantially cheaper and might change the trade-off in Section
4.3. *Entity handling*: at 10.9% of the test set, untranslated named entities
are the largest addressable category — a copy mechanism, a bilingual entity
lexicon, or oversampling entity-bearing sentences. *Contrastive evaluation*:
Section 5.2 showed corpus BLEU inverting on a specific grammatical constraint; a
targeted suite for the personal *a*, gender agreement, and subjunctive mood
would measure what BLEU averages away.

---

## References

Koehn, P. (2004). Statistical significance tests for machine translation
evaluation. *EMNLP*.

Kudo, T. and Richardson, J. (2018). SentencePiece: A simple and language
independent subword tokenizer and detokenizer for neural text processing.
*EMNLP: System Demonstrations*.

Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries.
*Workshop on Text Summarization Branches Out*.

Popović, M. (2015). chrF: character n-gram F-score for automatic MT evaluation.
*WMT*.

Post, M. (2018). A call for clarity in reporting BLEU scores. *WMT*.

Press, O. and Wolf, L. (2017). Using the output embedding to improve language
models. *EACL*.

Sennrich, R., Haddow, B., and Birch, A. (2016). Neural machine translation of
rare words with subword units. *ACL*.

Sennrich, R. and Zhang, B. (2019). Revisiting low-resource neural machine
translation: A case study. *ACL*.

Tiedemann, J. (2012). Parallel data, tools and interfaces in OPUS. *LREC*.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, Ł., and Polosukhin, I. (2017). Attention is all you need. *NeurIPS*.

Wu, Y. et al. (2016). Google's neural machine translation system: Bridging the
gap between human and machine translation. *arXiv:1609.08144*.

Xiong, R. et al. (2020). On layer normalization in the Transformer
architecture. *ICML*.

Zhang, B., Williams, P., Titov, I., and Sennrich, R. (2020). Improving massively
multilingual neural machine translation and zero-shot translation. *ACL*.
