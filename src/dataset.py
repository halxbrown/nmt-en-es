"""
Milestone 1, part 3: the PyTorch input pipeline.

Three ideas do most of the work here.

1. PRE-ENCODE ONCE, STORED FLAT.
   Tokenizing inside __getitem__ re-runs SentencePiece every epoch. We encode
   once and store the result as a single flat int32 array plus an offsets array
   (`RaggedArray`). Beyond speed, this avoids the classic PyTorch memory blow-up
   where a Python list of 100k objects gets refcount-touched by every DataLoader
   worker, defeating fork's copy-on-write.

2. DYNAMIC PADDING.
   The collate function pads to the longest sequence IN THE BATCH, not to a
   global maximum. Padding to a fixed 128 when the median sentence is 25 tokens
   means ~80% of your FLOPs are spent on <pad>.

3. LENGTH BUCKETING.
   Dynamic padding only helps if similar-length sentences land in the same
   batch. We shuffle, cut the shuffled order into "megabatches" of
   batch_size * pool_factor, sort each megabatch by source length, cut it into
   batches, then shuffle the batch order. You get near-homogeneous batches while
   keeping randomness across epochs.
"""


from __future__ import annotations

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # project root on path

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from config import CFG, LoaderConfig
from tokenizer import NMTTokenizer


# --------------------------------------------------------------------------
# Ragged storage
# --------------------------------------------------------------------------


class RaggedArray:
    """Variable-length int sequences stored as one flat array + offsets."""

    def __init__(self, flat: np.ndarray, offsets: np.ndarray):
        self.flat = flat
        self.offsets = offsets

    @classmethod
    def from_sequences(cls, seqs: list[list[int]]) -> "RaggedArray":
        lens = np.fromiter((len(s) for s in seqs), dtype=np.int64, count=len(seqs))
        offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
        np.cumsum(lens, out=offsets[1:])
        flat = (np.concatenate([np.asarray(s, dtype=np.int32) for s in seqs])
                if seqs else np.zeros(0, dtype=np.int32))
        return cls(flat, offsets)

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int) -> np.ndarray:
        return self.flat[self.offsets[i]:self.offsets[i + 1]]

    @property
    def lengths(self) -> np.ndarray:
        return np.diff(self.offsets)


# --------------------------------------------------------------------------
# Encoding
# --------------------------------------------------------------------------


def encode_corpus(
    pairs: list[tuple[str, str]],
    tok: NMTTokenizer,
    max_tokens: int,
) -> tuple[RaggedArray, RaggedArray]:
    """Encode pairs to id sequences, truncating the BODY so specials survive.

    Truncating after appending </s> would silently delete the end-of-sequence
    token on long sentences, and the model would never learn to stop.
    """
    src_seqs, tgt_seqs = [], []
    for s, t in pairs:
        s_ids = tok.sp.encode(s, out_type=int)[: max_tokens - 1]
        t_ids = tok.sp.encode(t, out_type=int)[: max_tokens - 2]
        src_seqs.append(s_ids + [tok.eos_id])
        tgt_seqs.append([tok.bos_id] + t_ids + [tok.eos_id])
    return RaggedArray.from_sequences(src_seqs), RaggedArray.from_sequences(tgt_seqs)


class TranslationDataset(Dataset):
    def __init__(self, src: RaggedArray, tgt: RaggedArray):
        assert len(src) == len(tgt), "src/tgt count mismatch"
        self.src, self.tgt = src, tgt

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i: int):
        return self.src[i], self.tgt[i]

    @property
    def src_lengths(self) -> np.ndarray:
        return self.src.lengths

    @property
    def bucket_lengths(self) -> np.ndarray:
        """Sort key for bucketing.

        Batches are padded to the max SOURCE length and, separately, the max
        TARGET length. Sorting on source alone leaves target-side padding
        unaddressed, since the two lengths only correlate. Summing them makes
        the sampler minimise total padded area across both tensors.
        """
        return self.src.lengths + self.tgt.lengths

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            src_flat=self.src.flat, src_off=self.src.offsets,
            tgt_flat=self.tgt.flat, tgt_off=self.tgt.offsets,
        )

    @classmethod
    def load(cls, path: Path) -> "TranslationDataset":
        z = np.load(path)
        return cls(RaggedArray(z["src_flat"], z["src_off"]),
                   RaggedArray(z["tgt_flat"], z["tgt_off"]))


# --------------------------------------------------------------------------
# Collate: dynamic padding + teacher-forcing shift
# --------------------------------------------------------------------------


def make_collate_fn(pad_id: int):
    def collate(batch):
        srcs = [b[0] for b in batch]
        tgts = [b[1] for b in batch]
        B = len(batch)
        S = max(len(x) for x in srcs)
        T = max(len(x) for x in tgts)

        src = torch.full((B, S), pad_id, dtype=torch.long)
        tgt = torch.full((B, T), pad_id, dtype=torch.long)
        for i, (s, t) in enumerate(zip(srcs, tgts)):
            src[i, : len(s)] = torch.from_numpy(s.astype(np.int64))
            tgt[i, : len(t)] = torch.from_numpy(t.astype(np.int64))

        # Teacher forcing: the decoder sees <s> y1 ... yn and must predict
        # y1 ... yn </s>. One tensor, two offset views.
        decoder_input = tgt[:, :-1].contiguous()
        labels = tgt[:, 1:].contiguous()

        return {
            "src": src,                                   # (B, S)
            "decoder_input": decoder_input,               # (B, T-1)
            "labels": labels,                             # (B, T-1)
            # PyTorch convention: True == "ignore this key position".
            # Verify this every time -- inverting it trains a model that
            # attends ONLY to padding and it will not obviously crash.
            "src_key_padding_mask": src.eq(pad_id),       # (B, S) bool
            "tgt_key_padding_mask": decoder_input.eq(pad_id),  # (B, T-1) bool
            "src_lengths": torch.tensor([len(x) for x in srcs], dtype=torch.long),
            "n_tgt_tokens": int(labels.ne(pad_id).sum()),  # for loss normalisation
        }

    return collate


# --------------------------------------------------------------------------
# Length-bucketed batch sampler
# --------------------------------------------------------------------------


class LengthGroupedBatchSampler(Sampler):
    """Yields lists of indices whose sequences have similar length."""

    def __init__(self, lengths: np.ndarray, batch_size: int, pool_factor: int = 50,
                 shuffle: bool = True, seed: int = 0, drop_last: bool = False):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.pool = batch_size * pool_factor
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._n_batches = len(self._build(epoch=0))

    def set_epoch(self, epoch: int) -> None:
        """Call this each epoch so bucket membership re-randomises."""
        self.epoch = epoch

    def _build(self, epoch: int) -> list[list[int]]:
        n = len(self.lengths)
        rng = np.random.default_rng(self.seed + epoch)
        order = rng.permutation(n) if self.shuffle else np.arange(n)

        batches: list[list[int]] = []
        for i in range(0, n, self.pool):
            chunk = order[i: i + self.pool]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            for j in range(0, len(chunk), self.batch_size):
                b = chunk[j: j + self.batch_size]
                if self.drop_last and len(b) < self.batch_size:
                    continue
                batches.append(b.tolist())

        if self.shuffle:
            batches = [batches[k] for k in rng.permutation(len(batches))]
        return batches

    def __iter__(self):
        yield from self._build(self.epoch)

    def __len__(self) -> int:
        return self._n_batches


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def make_dataloader(
    dataset: TranslationDataset,
    tok: NMTTokenizer,
    cfg: LoaderConfig = CFG.loader,
    train: bool = True,
    bucket: bool = True,
) -> DataLoader:
    collate = make_collate_fn(tok.pad_id)

    if bucket:
        sampler = LengthGroupedBatchSampler(
            dataset.bucket_lengths, cfg.batch_size,
            pool_factor=cfg.bucket_pool_factor,
            shuffle=train, seed=cfg.seed,
        )
        return DataLoader(
            dataset, batch_sampler=sampler, collate_fn=collate,
            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        )

    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=train,
        collate_fn=collate, num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory, drop_last=False,
    )


def padding_efficiency(loader: DataLoader, pad_id: int, max_batches: int = 200) -> dict:
    """Fraction of tensor cells that carry real tokens. Put this in the report."""
    real = cells = 0
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        for key in ("src", "decoder_input"):
            t = b[key]
            cells += t.numel()
            real += int(t.ne(pad_id).sum())
    return {
        "real_tokens": real,
        "tensor_cells": cells,
        "useful_fraction": real / max(cells, 1),
        "wasted_on_padding": 1 - real / max(cells, 1),
    }
