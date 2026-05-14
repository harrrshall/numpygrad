"""
data.py - the OpenWebText subset data loader.

Reads the uint16 token `.bin` memmaps produced by nanoGPT's
`data/openwebtext/prepare_subset.py` (train.bin ~5.6M tokens, val.bin ~286K).

`get_batch` mirrors nanoGPT's `train.py` `get_batch` exactly: one
`torch.randint` over `len(data) - block_size`, then gather `block_size`-token
windows. Because it consumes the global torch RNG identically to nanoGPT,
feeding it the same seeded RNG stream reproduces the baseline's exact batches.
It returns plain int64 NumPy arrays - the form `gpt.GPT` consumes; the caller
wraps them in `torch.from_numpy` for the PyTorch side.
"""
import os

import numpy as np
import torch

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "nanogpt", "data", "openwebtext")


def load_split(split):
    """Memmap the uint16 token .bin for `split` in {'train', 'val'}."""
    path = os.path.join(_DATA_DIR, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size, block_size):
    """One batch, exactly as nanoGPT's train.py `get_batch`: sample `batch_size`
    start positions with `torch.randint(len(data) - block_size, (batch_size,))`,
    then gather `block_size`-token input/target windows (target = input shifted
    by one). Returns (x, y) int64 NumPy arrays of shape (batch_size, block_size).

    Consumes exactly one `torch.randint` from the global torch RNG - identical
    to nanoGPT - so a faithfully-seeded call sequence reproduces its batches.
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = np.stack([np.asarray(data[i:i + block_size], dtype=np.int64) for i in ix])
    y = np.stack([np.asarray(data[i + 1:i + 1 + block_size], dtype=np.int64) for i in ix])
    return x, y
