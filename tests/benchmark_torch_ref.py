"""
benchmark_torch_ref.py - PyTorch reference for the Step 9 benchmark.

`benchmark.py` evaluates our from-scratch NumPy GPT-2 (loaded with real GPT-2
124M weights) and gets WikiText-103 ppl ~26.6 - lower than the GPT-2 paper's
37.50. That gap is *protocol*, not implementation: the paper used a different
(more conservative) evaluation than the strided-512 sliding window here.

This script proves that by running **PyTorch's GPT-2 124M** through the *exact
same* protocol, datasets, subsets and tokenizer as `benchmark.py`. It should
land on the same numbers our NumPy implementation produced - which, together
with the machine-precision parity from Steps 4-8, closes the loop:

    our NumPy impl  ==  PyTorch  ==/=  the paper's number (protocol differs)

Run:  .venv/bin/python tests/benchmark_torch_ref.py     (~3 min)
"""
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "nanogpt"))

from model import GPT as TorchGPT                                     # noqa: E402

# identical to benchmark.py
WIKITEXT_TOKENS = 18000
LAMBADA_EXAMPLES = 150
BLOCK_SIZE = 1024
STRIDE = BLOCK_SIZE // 2


def main():
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    print("loading datasets (same as benchmark.py) ...", flush=True)
    wt = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    wt_ids = np.array(enc.encode_ordinary("\n\n".join(wt["text"])), dtype=np.int64)[:WIKITEXT_TOKENS]
    lam = load_dataset("EleutherAI/lambada_openai", "default", split="test")
    lam_examples = [lam[i] for i in range(LAMBADA_EXAMPLES)]

    print("loading PyTorch GPT-2 124M ...", flush=True)
    g2 = TorchGPT.from_pretrained("gpt2")
    g2.eval()

    # --- WikiText-103 strided perplexity (same protocol as benchmark.py) ---
    nll_sum, n_tok, prev_end, begin, n_win = 0.0, 0, 0, 0, 0
    t0 = time.time()
    with torch.no_grad():
        while True:
            end = min(begin + BLOCK_SIZE, len(wt_ids) - 1)
            x = torch.from_numpy(wt_ids[None, begin:end].copy())
            y = torch.from_numpy(wt_ids[None, begin + 1:end + 1].copy())
            logits, _ = g2(x, y)                       # targets non-None -> all-position logits
            ce = F.cross_entropy(logits[0], y[0], reduction="none")
            trg_len = end - prev_end
            nll_sum += float(ce[-trg_len:].sum())
            n_tok += trg_len
            prev_end = end
            n_win += 1
            if end == len(wt_ids) - 1:
                break
            begin += STRIDE
    wt_ppl = float(np.exp(nll_sum / n_tok))
    print(f"  WikiText-103: ppl {wt_ppl:.2f}  ({n_tok} tokens, {n_win} windows, "
          f"{time.time() - t0:.0f}s)", flush=True)

    # --- LAMBADA (same protocol as benchmark.py) ---
    nll, ntok, correct, n = 0.0, 0, 0, 0
    t0 = time.time()
    with torch.no_grad():
        for ex in lam_examples:
            text = ex["text"].strip()
            if " " not in text:
                continue
            ctx, word = text.rsplit(" ", 1)
            ctx_ids = enc.encode_ordinary(ctx)
            tgt_ids = enc.encode_ordinary(" " + word)
            if not ctx_ids or not tgt_ids:
                continue
            ids = ctx_ids + tgt_ids
            if len(ids) > BLOCK_SIZE + 1:
                ids = ids[-(BLOCK_SIZE + 1):]
            n_tgt = min(len(tgt_ids), len(ids) - 1)
            x = torch.tensor(ids[:-1], dtype=torch.long)[None]
            y = torch.tensor(ids[1:], dtype=torch.long)[None]
            logits, _ = g2(x, y)
            ce = F.cross_entropy(logits[0], y[0], reduction="none")
            nll += float(ce[-n_tgt:].sum())
            ntok += n_tgt
            pred = logits[0].argmax(dim=-1)
            correct += int(torch.equal(pred[-n_tgt:], y[0, -n_tgt:]))
            n += 1
    lam_ppl = float(np.exp(nll / ntok))
    lam_acc = correct / n
    print(f"  LAMBADA: ppl {lam_ppl:.2f}  acc {lam_acc * 100:.2f}%  "
          f"({n} examples, {time.time() - t0:.0f}s)", flush=True)

    print("=" * 72)
    print(f"PyTorch GPT-2 124M (same protocol as benchmark.py):")
    print(f"  WikiText-103 ppl = {wt_ppl:.2f}")
    print(f"  LAMBADA ppl = {lam_ppl:.2f}   acc = {lam_acc * 100:.2f}%")
    print("=" * 72)
    print("compare to benchmark.py's NumPy-impl numbers: WikiText 26.57, "
          "LAMBADA ppl 21.67 / acc 38.00%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
