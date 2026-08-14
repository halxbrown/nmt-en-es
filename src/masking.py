"""
Mask construction. The padding masks are built in the collate function; this
module holds the causal mask and the verification helpers Milestone 2 asks for.

PyTorch conventions (get these wrong and nothing crashes -- the model just
quietly learns garbage):

  key_padding_mask : (B, S) bool. True  == this key position is PADDING, ignore.
  attn_mask        : (T, S) bool. True  == this (query, key) pair is FORBIDDEN.
                     (T, S) float. -inf == forbidden, 0.0 == allowed.

Note that bool True means "masked out" in both cases, which is the opposite of
the "1 = keep" convention used in a lot of tutorial code and in Hugging Face's
`attention_mask`. Pick one and assert it.
"""
from __future__ import annotations

import torch


def causal_mask(size: int, device=None) -> torch.Tensor:
    """(size, size) bool. True above the diagonal == cannot see the future."""
    return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)


def causal_mask_float(size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Additive -inf version, for manual scaled-dot-product attention."""
    m = torch.zeros(size, size, device=device, dtype=dtype)
    m.masked_fill_(causal_mask(size, device), float("-inf"))
    return m


def verify_masks(batch: dict, pad_id: int) -> list[str]:
    """Assertions that catch the three bugs that eat a whole afternoon."""
    checks = []
    src, dec, lab = batch["src"], batch["decoder_input"], batch["labels"]

    # 1. Padding masks flag exactly the pad positions, no more, no less.
    assert torch.equal(batch["src_key_padding_mask"], src.eq(pad_id))
    assert torch.equal(batch["tgt_key_padding_mask"], dec.eq(pad_id))
    checks.append("padding masks match pad positions exactly")

    # 2. Padding is a SUFFIX. If a real token appears after a pad, the collate
    #    wrote sequences in the wrong place and attention will be nonsense.
    for name, t in (("src", src), ("decoder_input", dec)):
        is_pad = t.eq(pad_id).int()
        # once padding starts it must never stop: diff can never go 1 -> 0
        assert (is_pad[:, 1:] - is_pad[:, :-1] >= 0).all(), f"{name}: pad not right-aligned"
    checks.append("padding is right-aligned (suffix only) in src and decoder_input")

    # 3. The teacher-forcing shift is exactly one position.
    assert dec.shape == lab.shape
    assert torch.equal(dec[:, 1:], lab[:, :-1]), "decoder_input/labels not offset by 1"
    checks.append("decoder_input and labels are offset by exactly one token")

    # 4. Causal mask shape agrees with the decoder length.
    T = dec.size(1)
    cm = causal_mask(T)
    assert cm.shape == (T, T)
    assert not cm.diagonal().any(), "a position must be allowed to see itself"
    assert not torch.tril(cm).any(), "the past must never be blocked"
    if T > 1:
        assert cm[0, 1:].all(), "position 0 must not see the future"
    checks.append(f"causal mask ({T}x{T}) blocks the future, allows diagonal and past")

    return checks
