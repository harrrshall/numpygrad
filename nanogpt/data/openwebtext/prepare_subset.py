"""
Faithful subset of nanoGPT's OpenWebText pipeline (data/openwebtext/prepare.py).

Identical to upstream in tokenizer (tiktoken gpt2 BPE), eot handling, and
output format (uint16 memmap .bin), but streams only N_TRAIN_DOCS / N_VAL_DOCS
documents from HuggingFace so it finishes in minutes on a laptop instead of
downloading 54 GB.
"""

import hashlib
import os
import sys

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

N_TRAIN_DOCS = int(os.environ.get("N_TRAIN_DOCS", 5000))
N_VAL_DOCS = int(os.environ.get("N_VAL_DOCS", 250))

# Skylion007/openwebtext is the canonical mirror used after the original
# `openwebtext` config was deprecated on the Hub.
HF_DATASET = os.environ.get("HF_DATASET", "Skylion007/openwebtext")

enc = tiktoken.get_encoding("gpt2")


def _doc_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stream_tokens(n_docs: int, skip: int, split_label: str):
    """Stream `n_docs` docs from HF (after skipping `skip`) and yield
    (tokens, sha1(text)) per doc.

    Mirrors upstream `process()` exactly: encode_ordinary + append eot.
    """
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    pbar = tqdm(total=n_docs, desc=f"tokenize {split_label}")
    n = 0
    for i, ex in enumerate(ds):
        if i < skip:
            continue
        if n >= n_docs:
            break
        text = ex["text"]
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)
        yield np.asarray(ids, dtype=np.uint16), _doc_hash(text)
        n += 1
        pbar.update(1)
    pbar.close()


def write_split(n_docs: int, skip: int, out_path: str, split_label: str):
    chunks = []
    hashes = set()
    total = 0
    for arr, h in stream_tokens(n_docs, skip, split_label):
        chunks.append(arr)
        hashes.add(h)
        total += arr.size

    assert len(hashes) == len(chunks), (
        f"{split_label}: {len(chunks) - len(hashes)} duplicate doc(s) "
        f"within split"
    )
    print(f"{split_label}: {len(chunks)} docs, {total:,} tokens -> {out_path}")
    mmap = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total,))
    idx = 0
    for arr in chunks:
        mmap[idx: idx + arr.size] = arr
        idx += arr.size
    mmap.flush()
    return hashes


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    train_hashes = write_split(
        N_TRAIN_DOCS, 0, os.path.join(here, "train.bin"), "train"
    )
    val_hashes = write_split(
        N_VAL_DOCS, N_TRAIN_DOCS, os.path.join(here, "val.bin"), "val"
    )

    overlap = train_hashes & val_hashes
    assert not overlap, (
        f"train/val overlap: {len(overlap)} docs, sample: "
        f"{list(overlap)[:3]}"
    )
    print(
        f"disjointness OK: train={len(train_hashes)} docs, "
        f"val={len(val_hashes)} docs, intersection=0"
    )
    print("done.", file=sys.stderr)
