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
and the larger model outperforms the smaller by +1.01 BLEU (p < 0.001). However,
the larger model under greedy decoding matches the smaller model under beam
search (24.82 vs 24.87 BLEU) at 3.5× the decoding throughput, suggesting that at
this scale additional compute is better invested in parameters than in search
width. We further identify a systematic length bias: the hypothesis/reference
length ratio is approximately 0.92 in every sentence-length bucket, costing an
estimated 2 BLEU to the brevity penalty alone.

---

## 1. Introduction and Motivation

### 1.1 Language pair and corpus

We translate English into Spanish using the `en-es` portion of OPUS-100
[Zhang et al., 2020], a multilingual corpus derived from OPUS
[Tiedemann, 2012]. Spanish was selected deliberately as a *high-resource*
pair despite the assignment's encouragement of low-resource settings. The
reasoning is methodological: with abundant clean data available, any performance
ceiling we observe can be attributed to model capacity, training budget, or
decoding strategy rather than to data scarcity. This makes the capacity ablation
in Section 4 interpretable in a way it would not be if the corpus were the
binding constraint.

### 1.2 Corpus characteristics

The `en-es` training split contains 1,000,000 sentence pairs. Inspection of the
data reveals that it is dominated by subtitle dialogue: representative training
sentences include *"- Chris."* and *"Can we dim the house lights a little bit
more?"*. This has three consequences that shape the entire evaluation.

First, the register is informal and conversational, not the formal register of
Europarl or news corpora. Second, sentences are short — the median source
sentence is 12 subword tokens against a mean of 18.3, indicating a strongly
right-skewed distribution. Third, and most consequentially for evaluation,
subtitle lines are *context-dependent fragments*. A reference translation may
legitimately contain material that the source sentence does not, because the
human subtitler had access to surrounding dialogue. We return to this in
Section 5.

### 1.3 Goals

Our objectives are (i) to implement the Transformer architecture from primitive
components rather than pre-built modules, (ii) to verify correctness
behaviourally rather than by inspection, (iii) to quantify the contributions of
model capacity and decoding search *separately*, with significance testing, and
(iv) to characterize failure modes concretely enough to motivate specific
improvements.

---

## 2. System Architecture

### 2.1 Data pipeline and filtering

Raw pairs pass through NFKC normalization (folding compatibility variants,
full-width punctuation, and non-breaking spaces) followed by eight filters.
Table 1 reports the yield.

**Table 1: Training-data filtering (1,000,000 input pairs)**

| Filter | Dropped | % of input |
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

Two entries deserve comment. The 8.41% dropped as duplicate-source reflects
OPUS's construction from overlapping subtitle releases; retaining multiple
targets for one source would let the model see contradictory supervision for
identical input. The 1.52% flagged as identical source and target is
untranslated English sitting in the Spanish column — a known OPUS artifact, and
copy-noise that actively teaches the model the wrong behaviour.

From the 835,357 retained pairs we sample 100,000 with a fixed seed (1337) to
meet the assignment's scale constraint. We use the *official* OPUS-100
validation and test splits (2,000 sentences each), which the corpus authors
guarantee not to overlap the training data — a guarantee a random split of our
own could not provide given the duplication rate observed above.

Critically, aggressive filtering is applied **only** to the training split.
Validation and test receive normalization and empty-line removal alone. Applying
length and noise filters to the test set would make it easier than the true
input distribution and inflate reported BLEU.

### 2.2 Tokenization

We train a single joint SentencePiece [Kudo and Richardson, 2018] BPE model
[Sennrich et al., 2016] of 16,000 pieces over the concatenation of both
language sides. A joint vocabulary is appropriate here because English and
Spanish share the Latin alphabet and a large cognate inventory
(*information*/*información*, *national*/*nacional*); shared subwords therefore
occupy one embedding row rather than two, numbers and names tokenize identically
on both sides, and — most importantly — the vocabulary can be tied three ways
(Section 2.3). The vocabulary size is matched to the corpus: a 32k vocabulary
over 3.79M subword tokens would leave the rarest rows badly undertrained.

Configuration details: `pad_id=0` (so a zero-initialized tensor is naturally
all-padding and mask logic reads as `ids == 0`), `split_digits=True` (so unseen
numbers can be copied digit by digit), `byte_fallback=True` (unknown characters
decompose to UTF-8 byte tokens rather than collapsing to `<unk>`), and
`nmt_nfkc` normalization. The tokenizer is trained on the **training split
only**; fitting it on validation or test text constitutes a mild but real leak.

**Table 2: Tokenizer health**

| Split | Fertility (subwords/word) | UNK rate |
|---|---:|---:|
| train.en | 1.49 | 0.0000% |
| train.es | 1.49 | 0.0000% |
| valid.en (unseen) | 1.53 | 0.0000% |
| valid.es (unseen) | 1.52 | 0.0000% |

The identical fertility across languages (1.49/1.49) indicates the joint
vocabulary is not skewed toward either side. The drift to 1.53/1.52 on unseen
text is small, indicating good generalization of the merge table. The zero UNK
rate is a direct consequence of `byte_fallback`; this matters for the error
analysis because it lets us distinguish *"rare token translated badly"* from
*"token was never representable."*

### 2.3 Model

Both presets share a pre-norm Transformer encoder–decoder with manually
implemented scaled dot-product attention:

```
Attention(Q,K,V) = softmax(QKᵀ / √d_head + M) V
```

where `M` is an additive mask filled with `finfo.min` at forbidden positions.
We use `finfo.min` rather than `-inf` because a fully masked softmax row over
`-inf` produces NaN, and because `-inf` overflows under fp16 autocast.

We implement attention manually rather than calling `nn.MultiheadAttention` for
two reasons: every module returns its attention weight matrix, so the attention
visualizer in Section 5 requires no forward hooks; and the mask application is
legible where a reader can verify it.

**Pre-norm residual blocks.** The original Transformer [Vaswani et al., 2017]
applies LayerNorm *after* the residual addition. That configuration produces
large gradients at the top of the stack early in training and effectively
requires the inverse-square-root warmup schedule to converge at all
[Xiong et al., 2020]. We place LayerNorm inside the residual branch, which is
substantially less sensitive to learning-rate choice — valuable when the compute
budget does not permit many failed runs. A final LayerNorm is applied after each
stack, without which the output scale grows with depth and the softmax
saturates.

**Three-way weight tying.** The source embedding, target embedding, and output
projection share a single (16000 × 512) matrix [Press and Wolf, 2017]. This is
legitimate only because the vocabulary is joint: token ID 4021 denotes the same
string in both languages. Tying removes approximately 8.2M parameters and acts
as a regularizer.

**Positional encoding.** Fixed sinusoidal encodings, chosen over learned
positions because the model must handle test sentences longer than any seen in a
100k-pair training set; sinusoids extrapolate, learned embeddings do not.
Embeddings are initialized with standard deviation `d_model^-0.5` and scaled by
`√d_model` before the positional encoding is added, so that the sinusoids do not
swamp token identity.

**Table 3: Capacity presets**

| | base | small |
|---|---:|---:|
| Encoder / decoder layers | 4 + 4 | 3 + 3 |
| `d_model` | 512 | 512 |
| Attention heads | 8 | 8 |
| `d_ff` | 2048 | 1024 |
| Dropout | 0.2 | 0.2 |
| Parameters (tied) | 37,619,712 | 23,965,696 |

Both presets are deliberately shallower than the paper's 6+6 base configuration
(~65M parameters). With 3.79M training subwords, a 65M-parameter model has
roughly seventeen parameters per token of training signal; Sennrich and Zhang
[2019] show that lower-resource NMT benefits from reduced depth and heavier
regularization rather than the published defaults. **Dropout is held constant at
0.2 across both presets**, so the ablation isolates capacity rather than
confounding it with regularization strength.

### 2.4 Input pipeline

Three design decisions govern throughput.

Sequences are tokenized once and stored as a flat `int32` array with an offsets
array, rather than as a Python list of lists. Beyond avoiding repeated
SentencePiece calls every epoch, this avoids the refcount-touching that defeats
`fork`'s copy-on-write in DataLoader workers.

Batches are padded dynamically to the longest sequence *in the batch*, not to a
global maximum. Given a median source length of 12 against a 128-token ceiling,
static padding would spend the overwhelming majority of attention FLOPs on
`<pad>`.

Dynamic padding only helps if similar-length sequences co-occur, so a custom
batch sampler shuffles indices, cuts them into megabatches of
`batch_size × 50`, sorts each megabatch by the *sum* of source and target
length, cuts batches, and shuffles batch order. Sorting on source length alone
leaves target-side padding unaddressed, since the two lengths correlate
imperfectly.

**Table 4: Padding efficiency (measured over 100 batches)**

| Batching strategy | Useful tensor cells | Wasted on padding |
|---|---:|---:|
| Random | 22.8% | 77.2% |
| Length-bucketed (src+tgt) | 78.0% | 22.0% |

Length bucketing yields a **3.42× reduction in wasted attention cells per
epoch**. Sorting on combined rather than source-only length accounted for 3.4
percentage points of this (74.6% → 78.0%).

---

## 3. Experimental Setup

### 3.1 Hyperparameters

**Table 5: Training configuration**

| Parameter | Value |
|---|---|
| Optimizer | AdamW, β=(0.9, 0.98), ε=1e-9 |
| Weight decay | 0.01 |
| Peak learning rate | 5e-4 |
| Schedule | 2,000-step linear warmup → cosine decay to 2% of peak |
| Batch size | 64 sentences (1,563 steps/epoch) |
| Loss | Cross-entropy, label smoothing 0.1, `ignore_index=pad` |
| Gradient clipping | 1.0 (global norm, after unscaling) |
| Precision | fp16 autocast with GradScaler |
| Epochs | 20 (early stopping patience 3, on validation loss) |
| Max sequence length | 128 subwords per side |
| Seed | 1337 |
| Hardware | NVIDIA Tesla T4 (Google Colab) |

The warmup exists for a narrower reason than in the original paper. With
pre-norm blocks, warmup is not required for stability; it serves only to prevent
Adam's second-moment estimates from being fixed by a handful of noisy early
batches. 2,000 steps is approximately 1.3 epochs.

Loss is averaged over non-padding target positions via `ignore_index`, which
keeps gradient scale comparable across batches whose token counts differ
several-fold under sentence-count batching.

Checkpoints are written to Google Drive after every epoch, carrying optimizer,
scheduler, and GradScaler state so that an interrupted session resumes rather
than restarting with a cold optimizer.

### 3.2 Correctness verification

Because a masking error in a Transformer degrades quality *without raising an
exception*, we verify correctness behaviourally rather than structurally. Four
structural assertions (padding masks match pad positions; padding is
right-aligned; decoder input and labels are offset by exactly one; the causal
mask blocks the strict upper triangle) are supplemented by three behavioural
tests:

**Causality.** Corrupting the target token at position *t* must leave every
prediction at positions < *t* bit-identical. Observed: maximum change
`0.000e+00` before position *t*, `1.040e+01` at and after. Information does not
flow backwards in time.

**Padding invariance.** Appending five `<pad>` columns to the source must leave
all logits unchanged. Observed: `0.000e+00` on CPU and `5.722e-06` on GPU. The
nonzero GPU value is cuBLAS selecting different reduction orders for different
tensor widths; floating-point addition is not associative. It is two orders of
magnitude inside the 1e-4 tolerance, and four orders below the scale of a real
leak.

**Decoder equivalence.** Beam search with k=1 and α=0 must reproduce greedy
decoding token-for-token, since with one beam there is nothing to compare and
the algorithm degenerates to argmax. It does. Additionally, beam search with k=5
finds sequences of higher model log-probability than greedy on 8/8 sampled
sentences (mean −40.86 → −20.70), and mean hypothesis length is non-decreasing
in α. Corpus BLEU cannot establish any of this: a subtly broken beam still
produces fluent output and a plausible score.

**Reproducibility.** The full Milestone-1 pipeline produces byte-identical
statistics on Windows/CPU (Python 3.11) and Linux/T4 (Python 3.12) — identical
filter counts, identical fertility, identical padding efficiency, and the same
sentence drawn into the same batch position. All randomness derives from
`seed=1337`.

### 3.3 Interpreting the initial loss

At initialization the model's loss is 10.2690 against `ln(16000) = 9.6803`, an
excess of +0.5886. This is expected rather than anomalous. Weight tying makes
output-projection row *j* identical to embedding row *j*, and the pre-norm
residual stream carries `embed(y_t)` to the output largely unchanged, so the
untrained model's strongest prediction is *the token it was just fed*. Measured
directly: the mean logit of the input token `y_t` is +8.032 against −0.005 for
all tokens. Since the label is the *next* token, loss necessarily begins above
uniform.

The diagnostic value lies in the opposite direction. Loss *below* `ln(V)` before
training would indicate that the model can already see its answer — that is, a
causal mask leak.

---

## 4. Results and Evaluation

All numbers in this section are computed on the 2,000-sentence held-out test
set, which was used neither for training nor for checkpoint selection. We report
sacreBLEU [Post, 2018] with signature
`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`; BLEU is not
comparable across tokenizations, so the bare number is not reproducible without
it. We additionally report chrF [Popović, 2015] and ROUGE-L [Lin, 2004].

### 4.1 Training behaviour

**Table 6: Training summary**

| | base | small |
|---|---:|---:|
| Epochs completed | 20 / 20 | 20 / 20 |
| Best validation loss | 3.4754 (epoch 19) | 3.5706 (epoch 19) |
| Validation perplexity | 32.31 | 35.54 |
| Seconds / epoch | 129–144 | 101–110 |

Neither model triggered early stopping. Validation loss improved monotonically
through epoch 19 and rose by only 0.0026 (base) and 0.0004 (small) at epoch 20.
**Neither model overfit**, despite base carrying roughly ten parameters per
training subword — dropout 0.2, label smoothing, and weight tying were
sufficient.

An important caveat: the learning rate reached its cosine floor (1.0e-5)
precisely as the curves flattened. We therefore *cannot* distinguish
convergence from the schedule ending. The evidence supports the claim that the
binding constraint was training budget rather than capacity, but not the
stronger claim that either model had converged.

### 4.2 Decoding strategy

**Table 7: Greedy vs. beam search (k=5, α=0.6) on the test set**

| Model | Decoding | BLEU | chrF | ROUGE-L | sent/s |
|---|---|---:|---:|---:|---:|
| base | greedy | 24.82 | 49.67 | 49.64 | 99.1 |
| base | beam k=5 | **25.88** | **50.49** | **50.84** | 18.9 |
| small | greedy | 23.99 | 48.80 | 49.00 | 140.2 |
| small | beam k=5 | 24.87 | 49.70 | 50.28 | 28.1 |

**Table 8: Paired-bootstrap significance (1,000 resamples)**

| Comparison | Δ BLEU | 95% CI | p |
|---|---:|---|---:|
| base: beam − greedy | +1.06 | [+0.69, +1.46] | < 0.001 |
| small: beam − greedy | +0.88 | [+0.54, +1.23] | < 0.001 |
| beam: base − small | +1.01 | [+0.50, +1.49] | < 0.001 |

A BLEU difference is meaningless without a variance estimate, so we test every
comparison with a paired bootstrap [Koehn, 2004]: the test set is resampled
with replacement, both systems are rescored on the *same* resample, and we count
how often the difference reverses. Because corpus BLEU is not the mean of
sentence BLEUs, per-sentence sufficient statistics (clipped n-gram matches,
n-gram totals, hypothesis and reference lengths) are cached once and
re-aggregated per resample. Our aggregation was validated against sacreBLEU to
six decimal places across four regimes spanning 1.57 to 67.76 BLEU.

All three comparisons are significant. Beam search delivers a real gain of
roughly +1 BLEU for both models, consistent with the +1–2 range reported in the
literature, and the improvement appears in chrF and ROUGE-L as well — the gain
is not a BLEU artifact.

We note that the beam gain is numerically larger for base (+1.06) than for small
(+0.88), but the confidence intervals overlap heavily; this difference is **not
established**. On the validation set the two gains were +1.27 and +1.26, whose
near-equality was coincidence rather than evidence of independence between
capacity and search.

### 4.3 The capacity/search trade-off

The most useful comparison in Table 7 is not between rows of the same model but
across them. **base under greedy decoding scores 24.82 BLEU at 99.1
sentences/second; small under beam search scores 24.87 at 28.1
sentences/second.** These are statistically indistinguishable in quality, and
the former is 3.5× faster.

At this scale, therefore, compute spent on parameters buys more than compute
spent on search width. A practitioner with a fixed latency budget should prefer
the larger model with greedy decoding to the smaller model with beam search.
This does not generalize beyond the operating point we measured, but it is a
concrete and testable conclusion from the ablation.

The counterargument is cost of a different kind: base requires +13.65M
parameters (+57%) for its +1.01 BLEU, and small decodes 1.4–1.5× faster at
matched decoding strategy. Which trade dominates depends on whether the
deployment constraint is memory or latency.

### 4.4 Performance by sentence length

**Table 9: BLEU by source length, beam k=5**

| Source words | n | base BLEU | base len ratio | small BLEU | small len ratio |
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
carries wide error bars and should be treated as suggestive rather than
established. Sinusoidal positional encodings and a 128-token ceiling that
truncated only 93 of 200,000 sequences (0.046%) both plausibly contribute.

**Very short sentences underperform medium ones.** The 0–9 bucket holds 55% of
the test set yet scores below 10–19 for both models. Two mechanisms are
plausible: short subtitle lines are context-dependent fragments whose correct
translation is underdetermined by the source alone, and BLEU is unstable on
short sequences, where a single wrong token eliminates a large fraction of the
available n-grams.

### 4.5 A systematic length bias

The `len ratio` column of Table 9 reveals a finding that the headline BLEU
conceals: the hypothesis/reference length ratio is approximately **0.92 in every
bucket**, not merely in the long ones. The model under-generates by roughly 8%
uniformly.

BLEU penalizes this directly. At a corpus ratio of 0.926 the brevity penalty is
`exp(1 − 1/0.926) ≈ 0.923`, so the reported 25.88 corresponds to an unpenalized
precision score near 28.0. **Approximately 2 BLEU is lost to length alone**,
before translation quality enters the calculation.

Since sequence log-probability is a sum of negative terms, beam search
intrinsically prefers shorter hypotheses; the GNMT length penalty
[Wu et al., 2016] at α=0.6 evidently does not fully correct for this. This
motivates the experiment in Section 6.

---

## 5. Error Analysis and Discussion

**Table 10: Heuristic error categories, beam k=5, base**

| Category | n | % of test set |
|---|---:|---:|
| Lexical / semantic (no structural flag) | 1,605 | 80.25 |
| Untranslated named entity | 218 | 10.90 |
| Truncation (len ratio < 0.6) | 122 | 6.10 |
| Over-generation (len ratio > 1.6) | 82 | 4.10 |
| Degenerate repetition | 6 | 0.30 |

Category rates for small are near-identical (79.65 / 11.40 / 6.20 / 4.00 /
0.45), indicating these failure modes are properties of the corpus and
tokenizer rather than of capacity. The detectors are heuristic and produce false
positives — a name flagged as untranslated may simply have been paraphrased in
the reference — so the following examples were read individually.

### 5.1 Failure 1: idiomatic multi-word expressions

> **SRC** Here are the sites of each of those that have taken place:
> **REF** Estos son los sitios en que cada Congreso ha tenido lugar:
> **base** Aquí están los sitios de cada uno de esos que **se han llevado a cabo**:
> **small** Aquí están los sitios de cada uno de esos que **han tomado lugar**:

This is the clearest capacity effect we observed. base renders *"have taken
place"* with the correct Spanish idiom *llevarse a cabo*; small produces
*tomar lugar*, a word-for-word calque that is not Spanish. Idiomatic
multi-word expressions require the model to suppress the compositional reading
in favour of a memorized mapping, and the additional layers and wider
feed-forward evidently support this.

Note also that the reference contains *"cada Congreso"* — a congress — which
appears nowhere in the source. This is the subtitle-fragment problem from
Section 1.2: the human subtitler had surrounding context the model does not.
Both systems are arguably *more* faithful to the given source than the reference
is, and BLEU penalizes them regardless. Sentence-level evaluation on subtitle
corpora systematically understates quality for this reason.

### 5.2 Failure 2: named entities and the personal *a*

> **SRC** I'm the man who killed Blackbeard.
> **REF** Sí. Soy el hombre que mató **a Barbanegra**.
> **base** Yo soy el hombre que mató **Blackbeard**.
> **small** Soy el hombre que mató **a Blackbeard**.

Two independent failures. Neither model localizes *Blackbeard* to *Barbanegra*;
both copy the English form. Given `byte_fallback`, this is not an OOV failure —
the token was representable — but a failure to learn a low-frequency
entity mapping from 100k pairs. At 10.9% of the test set, this is the largest
identifiable non-generic category.

More interesting is that **small produces the required personal *a* and base
omits it**. Spanish marks human direct objects with the preposition *a*; *mató
Blackbeard* is ungrammatical. The larger model, which wins on aggregate BLEU by
a statistically significant margin, fails on a specific grammatical constraint
that the smaller model satisfies. This is direct evidence that corpus-level BLEU
averages over per-phenomenon behaviour and can invert it. A targeted contrastive
evaluation set would be required to measure such constraints properly.

The reference's leading *"Sí."* is again bleed from an adjacent subtitle line.

### 5.3 Failure 3: syntactic recovery under beam search

> **SRC** I don't even remember what the fight was about.
> **REF** No recuerdo por qué fue la pelea.
> **base greedy** Ni siquiera recuerdo **lo que la pelea estaba**.
> **base beam** Ni siquiera recuerdo **lo de la pelea**.
> **small greedy** Ni siquiera recuerdo lo que **se había sobre eso**.
> **small beam** Ni siquiera recuerdo lo que la pelea estaba.

This illustrates the mechanism behind Table 8. base's greedy output calques the
English interrogative complement *"what the fight was about"*, stranding the
copula *estaba* at the end — locally plausible tokens that leave no grammatical
continuation. Beam search escapes to *"lo de la pelea"*, which is grammatical
though it loses the aboutness relation. Notice also the quality ladder: small's
beam output equals base's *greedy* output, consistent with the trade-off in
Section 4.3.

### 5.4 Attention behaviour

Cross-attention maps from the final decoder layer (averaged over heads) show a
clear diagonal band for short sentences, indicating that the model has learned
approximate monotonic word alignment. No alignment supervision appears anywhere
in the objective; this is an emergent consequence of the encoder–decoder
attention bottleneck. Off-diagonal mass concentrates where English and Spanish
word order diverges, notably around adjective–noun inversion.

### 5.5 Threats to validity

The test set is a single reference per source, so legitimate paraphrases are
penalized. The subtitle domain limits generalization to formal registers.
Length-bucket estimates for 50+ words rest on 37 sentences. The heuristic error
categories are unvalidated against human judgement. Finally, both models were
trained once with a single seed; the +1.01 BLEU capacity gap is significant
under resampling of the *test set* but we did not estimate variance across
training runs, which is typically comparable in magnitude.

---

## 6. Conclusion and Future Work

We built a pre-norm Transformer from PyTorch primitives and trained it on 100k
English–Spanish pairs, reaching 25.88 BLEU on a held-out test set with beam
search. Every architectural claim was verified behaviourally, and every
numerical comparison carries a paired-bootstrap confidence interval.

Three findings stand out. Beam search yields a significant ~+1 BLEU for both
capacity presets. The larger model beats the smaller by a significant +1.01
BLEU, but is matched in quality by *the larger model under greedy decoding at
3.5× throughput*, indicating that at this scale parameters are a better
investment than search width. And a uniform ~0.92 length ratio costs roughly 2
BLEU to the brevity penalty independently of sentence length.

**Immediate next step: length-penalty tuning.** Section 4.5 identifies ~2 BLEU
recoverable without retraining. We sweep α ∈ {0.0, 0.3, 0.6, 1.0, 1.4} on the
*validation* set and apply only the winning value to test — tuning a decoding
hyperparameter on test would contaminate it exactly as selecting a checkpoint on
test would. *[Insert results from `length_penalty_sweep_base.json`.]*

**Extended training.** Neither model converged before the cosine schedule
expired. A longer schedule at the same peak learning rate would establish
whether the 25.88 figure reflects the architecture or the budget.

**Incremental decoding.** Beam search currently re-runs the decoder over the
full prefix at every step, giving O(T²) total work. A key–value cache would make
beam search substantially cheaper and might change the trade-off in
Section 4.3.

**Entity handling.** At 10.9% of the test set, untranslated named entities are
the largest addressable category. Options include a copy mechanism, back-off to
a bilingual entity lexicon, or oversampling entity-bearing sentences.

**Contrastive evaluation.** Section 5.2 showed corpus BLEU inverting on a
specific grammatical constraint. A targeted test suite for the personal *a*,
gender agreement, and subjunctive mood would measure what BLEU averages away.

---

## References

Kudo, T. and Richardson, J. (2018). SentencePiece: A simple and language
independent subword tokenizer and detokenizer for neural text processing.
*EMNLP: System Demonstrations*.

Koehn, P. (2004). Statistical significance tests for machine translation
evaluation. *EMNLP*.

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

Zhang, B., Williams, P., Titov, I., and Sennrich, R. (2020). Improving
massively multilingual neural machine translation and zero-shot translation.
*ACL*.
