# EN → ES Neural Machine Translation with a Transformer

Custom PyTorch Transformer (Track A) trained on 100k parallel sentence pairs
from OPUS-100.

## Status

| Milestone | Status |
|---|---|
| M1 — data, tokenizer, DataLoader | done |
| M2 — model class, forward pass, masking | done |
| M3 — training loop, label smoothing, beam search | done |
| M4 — BLEU/ROUGE, error analysis | done |

## Results

Held-out test set, 2,000 sentences, never used for training or checkpoint
selection. sacreBLEU signature
`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`.

| Model | Params | Greedy | Beam k=5 | Beam + tuned α | chrF | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|
| base | 37.6M | 24.82 | 25.88 | **26.24** | 50.49 | 50.84 |
| small | 24.0M | 23.99 | 24.87 | 25.40 | 49.70 | 50.28 |

Every comparison carries a paired bootstrap (1,000 resamples):

| Comparison | Δ BLEU | 95% CI | p |
|---|---:|---|---:|
| base: beam − greedy | +1.06 | [+0.69, +1.46] | < 0.001 |
| small: beam − greedy | +0.88 | [+0.54, +1.23] | < 0.001 |
| beam: base − small | +1.01 | [+0.50, +1.49] | < 0.001 |
| base: tuned α − default | +0.36 | [+0.02, +0.63] | 0.021 |
| small: tuned α − default | +0.53 | [+0.29, +0.78] | < 0.001 |

Training: 20 epochs each on a Colab T4, ~2.3 min/epoch (base) and ~1.8 min
(small). Best validation loss 3.4754 and 3.5706, both at epoch 19; neither
model triggered early stopping.

## Running the translator locally

Training happens on Colab; inference runs fine on a laptop CPU. Three steps.

**1. Export a slim checkpoint (in Colab, after training).** `best.pt` carries
AdamW's two moment tensors per parameter plus scheduler and history — 430.7 MB
for base, where the weights alone are 143.6 MB (a 67% reduction). small goes
274.4 MB → 91.5 MB.

```python
!python export_model.py --preset base --checkpoint-dir "{CKPT_DIR}"
!cp "{CKPT_DIR}"/base/base_inference.pt "{DRIVE_ROOT}/"
```

**2. Download `base_inference.pt` from Drive into `models/`.**
The tokenizer is already in the repo (`artifacts/spm_enes.model`), which is
why that binary is version-controlled — the checkpoint is meaningless without
the exact vocabulary it was trained against.

**3. Verify:**

```bash
python setup_local.py
```

This checks packages, project files, tokenizer and checkpoint, then runs three
real translations. It reports precisely what is missing and how to fix it.
`translate.py` also validates that the checkpoint's embedding matrix matches the
tokenizer's vocabulary — a mismatch loads without error and then emits fluent
nonsense, so it is checked explicitly rather than left to be discovered.

In VS Code, **F5** offers three launch configurations: interactive translator,
single-sentence comparison, and the setup check.

## Quickstart

```bash
pip install -r requirements.txt

python run_milestone1.py --smoke   # ~20s, synthetic data, no download
python run_milestone1.py           # real run against opus-100 en-es
python run_milestone1.py --force   # rebuild data + tokenizer from scratch
python run_milestone2.py           # model build + masking verification
```

Milestones 3 and 4 need a GPU. Open `notebooks/nmt_full_pipeline.ipynb` in
Colab on a T4 — it runs the entire pipeline end to end, from data preparation
through training, evaluation, and export.

The real run downloads roughly 100 MB of Parquet, cleans 1M pairs down to
100k, trains a 16k joint BPE vocabulary, and caches the encoded corpus. It is
CPU-only and takes a few minutes — no GPU needed until Milestone 3.

## Layout

```
config.py                 all tunable settings in one dataclass tree

src/                      importable modules (training and inference share these)
  data_prep.py            download, normalise, filter, write parallel text
  tokenizer.py            joint SentencePiece BPE + wrapper
  dataset.py              ragged storage, dynamic-padding collate, bucketing
  masking.py              causal mask + structural mask assertions
  model.py                Transformer: manual attention, tied embeddings
  train.py                training loop, AMP, checkpointing, early stopping
  decode.py               greedy + batched beam search
  analysis.py             bootstrap significance, length buckets, attention maps

run_milestone1.py         M1: data pipeline + sanity report
run_milestone2.py         M2: model build + behavioural mask verification
run_milestone3.py         M3: train presets, compare greedy vs beam
run_milestone4.py         M4: held-out test evaluation + error analysis
verify_decoding.py        decoder correctness tests
sweep_length_penalty.py   tune length penalty on validation, apply once to test

translate.py              interactive EN->ES translator
setup_local.py            verify the local install can run the model
export_model.py           strip optimizer state for a portable checkpoint

notebooks/                nmt_full_pipeline.ipynb — all milestones, one notebook
reports/                  report draft
models/                   downloaded inference checkpoints (gitignored)
artifacts/                tokenizer model, encoded caches, stats JSON
data/processed/           train/valid/test .en and .es plain text
.vscode/                  launch configs (F5 runs the translator)
```

## Design decisions worth defending in the report

**Joint 16k BPE vocabulary.** EN and ES share the Latin alphabet and a large
cognate inventory, so one shared vocabulary lets cognates share an embedding row
and permits tying the encoder embedding, decoder embedding and output projection
into a single matrix. At 100k training pairs, parameter economy matters more
than vocabulary precision. 16k is sized to the corpus: a 32k vocabulary over
~4M subwords leaves the rare rows badly undertrained.

**`pad_id = 0`.** SentencePiece defaults to `pad_id = -1` (disabled). Setting it
to 0 explicitly means a zero-initialised tensor is naturally all-padding and
mask logic reads as `ids == 0`.

**`byte_fallback = True`.** Unknown characters decompose into UTF-8 byte tokens
instead of collapsing to `<unk>`, driving the practical UNK rate to zero. In
Milestone 4 this lets the error analysis distinguish "rare token translated
badly" from "token was never representable".

**Tokenizer trained on the training split only.** Fitting it on validation or
test text is a mild but real leak.

**Test set is cleaned minimally.** Aggressive length and noise filtering is
applied to training data only. Filtering the test set makes it easier than the
real distribution and inflates BLEU.

**Official opus-100 validation/test splits.** These are guaranteed by the corpus
authors not to overlap the training data, which a random split cannot promise
given how much duplication OPUS contains.

**Dynamic padding + length bucketing.** The collate function pads to the longest
sequence in the batch, and the sampler groups similar-length sequences by
sorting within shuffled megabatches, keyed on the *sum* of source and target
length so both tensors are packed. Measured on the real corpus: useful tensor
occupancy rises from 22.8% to 78.0%, a 3.42× reduction in wasted attention
cells per epoch. Sorting on combined rather than source-only length accounted
for 3.4 points of that (74.6% → 78.0%).

**Mask convention.** `True` means "ignore this position" for both
`key_padding_mask` and boolean `attn_mask`, matching PyTorch. This is the
opposite of Hugging Face's `attention_mask`, where 1 means "keep". Inverting it
does not crash — the model simply attends only to padding — so
`src/masking.py::verify_masks` asserts it on every sanity run.

## Reproducibility

All seeds derive from `config.py` (`seed = 1337`), covering the training
subsample, the tokenizer's input sampling, and the bucketing sampler's shuffle.
Call `sampler.set_epoch(epoch)` each epoch during training so bucket membership
re-randomises while staying deterministic.


## Milestone 2 notes

**Manual scaled dot-product attention** rather than `nn.MultiheadAttention`.
Every attention module returns its weight matrix, so the Milestone 4 attention
visualiser needs no forward hooks, and the mask application is readable.

**Pre-norm residual blocks.** The 2017 paper normalises after the residual add,
which needs the inverse-sqrt warmup schedule to converge. Pre-norm is far less
sensitive to learning rate — worth the small quality cost when GPU hours are
limited. Flip `pre_norm=False` in `config.py` to reproduce the original.

**Three-way weight tying** (source embedding, target embedding, output
projection) is only legitimate because the vocabulary is joint: token id 4021
denotes the same string in both languages. It removes ~8M parameters and
regularises a model that would otherwise memorise 100k pairs.

**Why initial loss exceeds ln(V).** Tying makes generator row *j* identical to
embedding row *j*, and the pre-norm residual stream carries `embed(y_t)` to the
output unchanged. The untrained model's strongest prediction is therefore the
token it was just fed, while the label is the *next* token — so loss starts
above uniform by construction. `run_milestone2.py` measures the bias directly.
Loss *below* ln(V) at initialisation is the alarming case: it means the causal
mask leaks.

**Behavioural mask verification.** Milestone 1 checked mask *shapes*. Milestone 2
checks mask *effects*: corrupting a target token must leave every earlier
prediction bit-identical, and appending source padding must leave all logits
unchanged. Measured: causal `0.000e+00` before the corrupted position against
`1.040e+01` after; padding `0.000e+00` on CPU and `5.722e-06` on GPU. The
nonzero GPU value is cuBLAS choosing different reduction orders for different
tensor widths — floating-point addition is not associative — and sits six orders
of magnitude below the scale of a real leak. A mask can have the right shape and
still be applied to the wrong axis, which structural assertions cannot catch.

**Overfitting a single batch** is the last gate before a real training run. A
correct seq2seq model memorises one batch quickly; if loss plateaus there, more
data and more epochs will not help.


## Milestone 3 notes

**Two capacity presets, trained and compared.** `base` is 37.6M parameters
(4+4 layers, d_ff 2048); `small` is 24.0M (3+3 layers, d_ff 1024). Dropout is
held at 0.2 for both, so the ablation isolates capacity rather than confounding
it with regularisation. 37.6M parameters against 3.79M training subwords is
roughly ten parameters per token of signal, so the comparison is a real
question, not a formality.

**Schedule.** Linear warmup (2000 steps, ~1.3 epochs) then cosine decay to 2% of
peak. The paper's inverse-sqrt schedule exists largely to stabilise Post-LN
training; with Pre-LN, warmup only needs to stop Adam's second-moment estimates
being fixed by a few noisy early batches.

**Mixed precision.** fp16 autocast roughly doubles T4 throughput. The attention
mask fills with `finfo.min` rather than `-inf` specifically so it survives fp16
without producing NaN.

**Checkpoint every epoch to Drive.** Colab kills sessions without warning.
`last.pt` carries optimizer, scheduler and GradScaler state, so a resumed run
continues rather than restarting with a cold optimizer. Re-running the training
cell resumes automatically.

**Decoding.** Greedy and batched beam search (k=5) with the GNMT length penalty
`((5+len)/6)^alpha`, alpha=0.6. Without a length penalty, beam search
systematically truncates: sequence log-probability is a sum of negative terms,
so shorter hypotheses always score higher.

**Decoder verification** (`verify_decoding.py`) — BLEU cannot tell you whether
beam search is correct, because a subtly broken beam still produces fluent
output and a plausible score. Three independent checks instead:

1. `beam(k=1, alpha=0)` must reproduce greedy token-for-token. With one beam
   there is nothing to compare, so it degenerates to argmax; any difference
   means the reordering, scoring or termination logic is wrong.
2. `beam(k=5)` must find sequences of higher model log-probability than greedy.
   That is beam search's entire purpose.
3. Mean hypothesis length must be non-decreasing in alpha.

Measured on a length-diverse probe (source lengths 9–128 subwords): base
improves mean sequence log-probability from −15.005 to −11.230, strictly better
on 4/8 sentences and never worse; small from −25.003 to −11.485, strictly better
on 2/8. That beam is only *strictly* better on some sentences is expected — on
the rest greedy already found the highest-probability sequence.

**A test that passed vacuously.** The first version of this script drew its
probe from the first batch of a length-bucketed loader. Because the sampler
sorts by length, that returned the *eight shortest* validation sentences — mean
hypothesis length 3.2 subwords. On inputs that short, greedy and beam produce
identical output and no length penalty can lengthen anything, so checks 2 and 3
reported PASS while testing nothing. Sentence selection now spans the length
distribution. A test that passes for the wrong reason is worse than no test.

**Order restoration.** `translate_corpus` sorts by length for efficient batching
and then inverts the permutation. Forgetting the inversion misaligns hypotheses
against references and yields a near-zero BLEU that looks exactly like a model
failure.

## Milestone 4 notes

**Test set, not validation.** Milestone 3's numbers came from the validation
set, which selected the checkpoint, so they are optimistically biased. All final
results are computed on `test`, untouched until this point.

**Significance testing.** A BLEU gap means nothing without it. We use a paired
bootstrap (Koehn, 2004): resample the test set with replacement, rescore both
systems on the *same* resample, and count how often the gap reverses. Since
corpus BLEU is not the mean of sentence BLEUs, per-sentence sufficient
statistics (clipped n-gram matches, totals, lengths) are cached once and
re-aggregated per resample. The implementation was validated to match sacreBLEU
to six decimal places across four noise regimes from 1.57 to 67.76 BLEU.

**Length-bucketed BLEU.** Corpus BLEU is dominated by whatever length dominates
the corpus, and OPUS-100 en-es is subtitle dialogue with a median of 12
subwords. A headline score can hide near-total failure on long sentences, so
results are reported per bucket with the hypothesis/reference length ratio
alongside. The measured ratio turned out to be ≈0.92 in *every* bucket, not just
the long ones — a uniform under-generation bias rather than length-specific
truncation.

**sacreBLEU signature** is recorded with every score
(`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`). BLEU is not
comparable across different tokenizations, so the bare number is not
reproducible without it.

**Error categories** (truncation, over-generation, repetition, untranslated
entity) are heuristic detectors meant to surface candidates. The report quotes
sentences that were actually read, not whatever the classifier labelled.

**Attention maps** come free because attention is implemented manually and every
module returns its weight matrix — no forward hooks needed.

## Length-penalty tuning notes

`sweep_length_penalty.py` sweeps the GNMT exponent α on the **validation** set
and applies only the winning value to test. Tuning a decoding hyperparameter by
test BLEU would contaminate the test set exactly as selecting a checkpoint on
test would.

Two findings worth recording.

**The BLEU-optimal length ratio is ≈0.98, not 1.0.** The sweep crosses the
optimum: at α=2.6 the test ratio is 0.979 for 26.33 BLEU, at α=4.0 it reaches
0.991 for 26.24. Driving hypotheses to reference length is achievable but costs
more n-gram precision than it recovers in brevity penalty.

**A monotonically improving tuning curve is not evidence that the largest value
is best.** base's validation BLEU rose at every α from 0.6 to 4.0, but the
spread above α≈1.8 was only 0.38 BLEU on 1,000 sentences — within sampling
noise. The validation argmax (α=4.0) transferred *worse* to test than α=2.6 did.
We report the α=4.0 result because that is what the protocol selects; choosing
26.33 instead would be selection on the test set.

## Future work

**Incremental decoding with a KV cache.** Each beam step re-runs the decoder
over the whole prefix, O(T²) overall. Not the bottleneck at these lengths, and
the simpler code is far easier to verify.

**A larger tuning set.** Length-penalty selection is noise-limited at 1,000
validation sentences. Tuning on the full 2,000, or averaging over bootstrap
resamples rather than taking a point argmax, would make it reproducible.

**Extended training.** Neither model converged before the cosine schedule
expired — the learning rate hit its floor exactly as the curves flattened, so
convergence cannot be distinguished from the schedule ending.

**Entity handling.** Untranslated named entities are 10.9% of the test set, the
largest addressable error category. Options: a copy mechanism, a bilingual
entity lexicon, or oversampling entity-bearing sentences.

**Contrastive evaluation.** Corpus BLEU inverted on at least one grammatical
constraint (the smaller model produced the required personal *a* where the
larger omitted it). A targeted suite would measure what BLEU averages away.
