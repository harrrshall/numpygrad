"""
Freeze a parity fixture for the Phase 4 NumPy implementation.

Saves into fixtures/parity_batch.pt a dict containing:
    x, y           : one (batch_size, block_size) int64 batch from val.bin
    state_dict     : the trained ckpt's model state_dict (CPU tensors)
    model_args     : the GPTConfig kwargs the ckpt was trained with
    config         : the full training config dict
    meta           : provenance — how (x,y) was sampled, hashes, dtype
                     so the NumPy side can reproduce/verify

Usage on the NumPy side (Phase 4):
    fx = torch.load('fixtures/parity_batch.pt', weights_only=False)
    forward_np = numpy_gpt(fx['state_dict'], fx['model_args'])(fx['x'])
    assert allclose(forward_np, pytorch_forward, atol=1e-5)
"""

import hashlib
import os

import numpy as np
import torch

CKPT_PATH = "out-owt-baseline/ckpt.pt"
VAL_BIN = "data/openwebtext/val.bin"
OUT_PATH = "fixtures/parity_batch.pt"

# Sampling params for the fixture. Use the same shapes the baseline trained at.
BATCH_SIZE = 12
BLOCK_SIZE = 64
SAMPLE_SEED = 20260513  # distinct from training seed (1337) so this batch
                        # isn't the literal first training batch.


def _sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def main():
    assert os.path.exists(CKPT_PATH), f"missing {CKPT_PATH}"
    assert os.path.exists(VAL_BIN), f"missing {VAL_BIN}"

    # 1. sample one (x, y) batch from val.bin using a pinned torch seed.
    data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    g = torch.Generator().manual_seed(SAMPLE_SEED)
    ix = torch.randint(
        len(data) - BLOCK_SIZE, (BATCH_SIZE,), generator=g
    )
    x = torch.stack([
        torch.from_numpy(np.asarray(data[i: i + BLOCK_SIZE], dtype=np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(
            np.asarray(data[i + 1: i + 1 + BLOCK_SIZE], dtype=np.int64)
        )
        for i in ix
    ])
    assert x.shape == (BATCH_SIZE, BLOCK_SIZE) == y.shape
    assert x.dtype == y.dtype == torch.int64

    # 2. load the trained checkpoint.
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    # nanoGPT sometimes prefixes keys with `_orig_mod.` when torch.compile is
    # used. Strip for the parity fixture (we ran with compile=False so this
    # should be a no-op, but make it defensive for the NumPy loader).
    prefix = "_orig_mod."
    state_dict = {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }

    # 3. provenance / meta — lets the NumPy side cross-check.
    sd_bytes = b"".join(
        v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        for _, v in sorted(state_dict.items())
    )
    meta = {
        "sample_seed": SAMPLE_SEED,
        "sampling_indices": ix.tolist(),
        "x_sha1": _sha1(x.numpy().tobytes()),
        "y_sha1": _sha1(y.numpy().tobytes()),
        "state_dict_sha1": _sha1(sd_bytes),
        "ckpt_iter": ckpt.get("iter_num"),
        "ckpt_best_val_loss": float(ckpt.get("best_val_loss", float("nan"))),
        "torch_version": torch.__version__,
    }

    fixture = {
        "x": x,
        "y": y,
        "state_dict": state_dict,
        "model_args": ckpt["model_args"],
        "config": ckpt["config"],
        "meta": meta,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    torch.save(fixture, OUT_PATH)

    # 4. report.
    n_params = sum(v.numel() for v in state_dict.values())
    sd_size_mb = sum(
        v.element_size() * v.numel() for v in state_dict.values()
    ) / 1e6
    print(f"wrote {OUT_PATH}")
    print(f"  x: {tuple(x.shape)} {x.dtype}  sha1={meta['x_sha1'][:12]}…")
    print(f"  y: {tuple(y.shape)} {y.dtype}  sha1={meta['y_sha1'][:12]}…")
    print(
        f"  state_dict: {len(state_dict)} tensors, "
        f"{n_params:,} params, {sd_size_mb:.2f} MB  "
        f"sha1={meta['state_dict_sha1'][:12]}…"
    )
    print(f"  model_args: {ckpt['model_args']}")
    print(f"  ckpt iter: {meta['ckpt_iter']}, "
          f"best_val_loss: {meta['ckpt_best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
