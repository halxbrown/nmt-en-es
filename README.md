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

## Quickstart

```bash
pip install -r requirements.txt

python run_milestone1.py --smoke   # ~20s, synthetic data, no download
python run_milestone1.py           # real run against opus-100 en-es
python run_milestone1.py --force   # rebuild data + tokenizer from scratch
python run_milestone2.py           # model build + masking verification
```

The real run downloads roughly 200 MB of Parquet, cleans ~1M pairs down to
100k, trains a 16k joint BPE vocabulary, and caches the encoded corpus. It is
CPU-only and takes a few minutes — no GPU needed until Milestone 3.

## Layout

```
config.py                 all tunable settings in one dataclass tree
run_milestone1.py         driver + sanity report
src/data_prep.py          download, normalise, filter, write parallel text
src/tokenizer.py          joint SentencePiece BPE + wrapper
src/dataset.py            ragged storage, dynamic-padding collate, bucketing
src/masking.py            causal mask + structural mask assertions
src/model.py              Transformer: manual attention, tied embeddings
src/train.py              training loop, AMP, checkpointing, early stopping
src/decode.py             greedy + batched beam search
run_milestone2.py         model build + behavioural mask verification
run_milestone3.py         train presets, compare greedy vs beam
verify_decoding.py        decoder correctness tests
src/analysis.py           bootstrap significance, length buckets, attention maps
run_milestone4.py         held-out test evaluation + error analysis
notebooks/                Colab notebook (T4 + Drive checkpointing)
artifacts/                tokenizer model, encoded caches, stats JSON
data/processed/           train/valid/test .en and .es plain text
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
sorting within shuffled megabatches. On the smoke corpus this moves useful
tensor occupancy from 53% to 89%; expect a similar gap on real data. Report the
measured numbers from `artifacts/milestone1_stats.json`.

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
bit-identical. Both currently return exactly 0.0 change. A mask can have the
right shape and still be applied to the wrong axis, which structural assertions
cannot catch.

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

**Order restoration.** `translate_corpus` sorts by length for efficient batching
and then inverts the permutation. Forgetting the inversion misaligns hypotheses
against references and yields a near-zero BLEU that looks exactly like a model
failure.

## Future work

Incremental decoding with a KV cache. Each beam step currently re-runs the
decoder over the whole prefix, which is O(T^2) overall. At these lengths it is
not the bottleneck, and the simpler code is far easier to verify.


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
alongside — a ratio well under 1.0 in the long buckets is direct evidence of
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
