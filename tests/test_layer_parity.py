"""
test_layer_parity.py - Step 3 gate: layer-by-layer PyTorch parity.

For every layer in layers.py: build the equivalent layer in PyTorch, copy the
*same* weights across, run forward + backward on the *same* input with the
*same* upstream gradient, and assert

    forward outputs    match to  1e-5  absolute
    parameter grads    match to  1e-4  relative
    input gradient     match to  1e-4  relative   (N/A for Embedding - integer ids)

This is layer-by-layer parity, not whole-model parity: if a layer breaks, this
says exactly which one. Order: Linear, LayerNorm, Embedding, MLP, Attention
(single-head then multi-head), TransformerBlock.

PyTorch is used only as the reference oracle (not in the model). It runs in
float64 here so any mismatch is a real bug, not float32 noise. The torch
reference modules mirror nanoGPT's model.py exactly (manual causal attention,
exact-erf GeLU via F.gelu, pre-LN block).

Run:  .venv/bin/python tests/test_layer_parity.py
Writes: tests/layer_parity.md
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)          # parity reference at full precision

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import layers                                   # noqa: E402
from tensor import Tensor                       # noqa: E402

FWD_ATOL = 1e-5
GRAD_RTOL = 1e-4
RESULTS_MD = os.path.join(_HERE, "layer_parity.md")


# --------------------------------------------------------------------------
# PyTorch reference modules - mirror nanoGPT's model.py
# --------------------------------------------------------------------------

class TorchMLP(nn.Module):
    def __init__(self, n_embd, bias=True):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x)))     # F.gelu default = exact erf


class TorchMHA(nn.Module):
    """nanoGPT CausalSelfAttention, manual attention path."""

    def __init__(self, n_embd, n_head, bias=True):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

    def forward(self, x):
        B, Tn, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        d = C // self.n_head
        q = q.view(B, Tn, self.n_head, d).transpose(1, 2)
        k = k.view(B, Tn, self.n_head, d).transpose(1, 2)
        v = v.view(B, Tn, self.n_head, d).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        mask = torch.triu(torch.ones(Tn, Tn, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float('-inf'))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, Tn, C)
        return self.c_proj(y)


class TorchBlock(nn.Module):
    def __init__(self, n_embd, n_head, bias=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd, eps=1e-5)
        self.attn = TorchMHA(n_embd, n_head, bias)
        self.ln_2 = nn.LayerNorm(n_embd, eps=1e-5)
        self.mlp = TorchMLP(n_embd, bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# --------------------------------------------------------------------------
# weight sync + comparison helpers
# --------------------------------------------------------------------------

# my parameter-name suffix -> torch parameter-name suffix
_TORCH_NAME = {'W': 'weight', 'b': 'bias', 'gamma': 'weight', 'beta': 'bias',
               'weight': 'weight'}
# suffixes that are stored (in,out) by us but (out,in) by torch. ONLY Linear
# weights (`W`); transpose must be decided by type, not shape - a square Linear
# (e.g. c_proj: n_embd -> n_embd) has identical shapes either way.
_TRANSPOSE = {'W'}


def _torch_param(torch_module, my_name):
    *prefix, last = my_name.split('.')
    tname = '.'.join(prefix + [_TORCH_NAME[last]])
    return dict(torch_module.named_parameters())[tname]


def randomize(layer):
    """Give every parameter a fresh, non-degenerate value (and clear stale grads)."""
    for p in layer.parameters().values():
        p.data = np.random.randn(*p.data.shape) * 0.5
        p.grad = None


def sync_weights(my_layer, torch_module):
    """Copy my_layer's parameters into torch_module, transposing Linear weights
    (we store (in,out), torch stores (out,in))."""
    for my_name, my_t in my_layer.parameters().items():
        tp = _torch_param(torch_module, my_name)
        val = my_t.data.T if my_name.split('.')[-1] in _TRANSPOSE else my_t.data
        with torch.no_grad():
            tp.copy_(torch.from_numpy(np.ascontiguousarray(val)))


def torch_grad_for(torch_module, my_name):
    """The torch gradient for the param matching `my_name`, in MY layout."""
    tp = _torch_param(torch_module, my_name)
    g = tp.grad.detach().numpy()
    return g.T if my_name.split('.')[-1] in _TRANSPOSE else g


def cmp_forward(my_out, torch_out):
    a = np.asarray(my_out, dtype=np.float64)
    b = torch_out.detach().numpy()
    if a.shape != b.shape:
        return float('inf'), False
    diff = float(np.abs(a - b).max())
    return diff, diff < FWD_ATOL


def cmp_grad(my_grad, torch_grad):
    a = np.asarray(my_grad, dtype=np.float64)
    b = np.asarray(torch_grad, dtype=np.float64)
    if a.shape != b.shape:
        return float('inf'), False
    diff = float(np.abs(a - b).max())
    denom = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-12)
    rel = diff / denom
    return rel, rel < GRAD_RTOL


def run_layer(name, my_layer, torch_module, x_np, ids_input=False):
    """Forward + backward parity for one layer. Returns list of result rows."""
    sync_weights(my_layer, torch_module)

    if ids_input:
        my_out = my_layer(x_np)                                  # integer ids
        torch_out = torch_module(torch.tensor(x_np, dtype=torch.long))
        xt = xt_torch = None
    else:
        xt = Tensor(x_np, requires_grad=True)
        xt_torch = torch.tensor(x_np, requires_grad=True)
        my_out = my_layer(xt)
        torch_out = torch_module(xt_torch)

    rows = []
    diff, ok = cmp_forward(my_out.data, torch_out)
    rows.append((name, 'forward', 'abs', diff, ok))

    # backward with the SAME upstream gradient on both sides
    g = np.random.randn(*my_out.shape)
    (my_out * Tensor(g)).sum().backward()
    (torch_out * torch.tensor(g)).sum().backward()

    if not ids_input:
        rel, ok = cmp_grad(xt.grad, xt_torch.grad.detach().numpy())
        rows.append((name, 'd_input', 'rel', rel, ok))

    for my_name, my_t in my_layer.parameters().items():
        rel, ok = cmp_grad(my_t.grad, torch_grad_for(torch_module, my_name))
        rows.append((name, f'd[{my_name}]', 'rel', rel, ok))
    return rows


# --------------------------------------------------------------------------
# per-layer tests, in the prescribed order
# --------------------------------------------------------------------------

def test_linear():
    np.random.seed(1)
    n_in, n_out = 8, 16
    my = layers.Linear(n_in, n_out, bias=True)
    randomize(my)
    return run_layer("Linear", my, nn.Linear(n_in, n_out, bias=True),
                     np.random.randn(4, 3, n_in))


def test_linear_no_bias():
    np.random.seed(8)
    n_in, n_out = 8, 16
    my = layers.Linear(n_in, n_out, bias=False)
    randomize(my)
    return run_layer("Linear (no bias)", my, nn.Linear(n_in, n_out, bias=False),
                     np.random.randn(4, 3, n_in))


def test_layernorm():
    np.random.seed(2)
    n_dim = 16
    my = layers.LayerNorm(n_dim, bias=True, eps=1e-5)
    randomize(my)
    return run_layer("LayerNorm", my, nn.LayerNorm(n_dim, eps=1e-5),
                     np.random.randn(4, 3, n_dim))


def test_embedding():
    np.random.seed(3)
    n_vocab, n_dim = 20, 8
    my = layers.Embedding(n_vocab, n_dim)
    randomize(my)
    ids = np.random.randint(0, n_vocab, size=(4, 3))
    return run_layer("Embedding", my, nn.Embedding(n_vocab, n_dim), ids, ids_input=True)


def test_mlp():
    np.random.seed(4)
    n_embd = 8
    my = layers.MLP(n_embd, bias=True)
    randomize(my)
    return run_layer("MLP", my, TorchMLP(n_embd, bias=True),
                     np.random.randn(4, 3, n_embd))


def test_attention_single_head():
    np.random.seed(5)
    n_embd, n_head = 8, 1
    my = layers.MultiHeadAttention(n_embd, n_head, bias=True)
    randomize(my)
    return run_layer("Attention (1 head)", my, TorchMHA(n_embd, n_head, bias=True),
                     np.random.randn(2, 5, n_embd))


def test_attention_multi_head():
    np.random.seed(6)
    n_embd, n_head = 16, 4
    my = layers.MultiHeadAttention(n_embd, n_head, bias=True)
    randomize(my)
    return run_layer("Attention (4 heads)", my, TorchMHA(n_embd, n_head, bias=True),
                     np.random.randn(2, 5, n_embd))


def test_transformer_block():
    np.random.seed(7)
    n_embd, n_head = 16, 4
    my = layers.TransformerBlock(n_embd, n_head, bias=True, eps=1e-5)
    randomize(my)
    return run_layer("TransformerBlock", my, TorchBlock(n_embd, n_head, bias=True),
                     np.random.randn(2, 5, n_embd))


TESTS = [
    test_linear,
    test_linear_no_bias,
    test_layernorm,
    test_embedding,
    test_mlp,
    test_attention_single_head,
    test_attention_multi_head,
    test_transformer_block,
]


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def main():
    print(f"layer parity vs PyTorch (float64)  |  forward < {FWD_ATOL:.0e} abs   "
          f"grads < {GRAD_RTOL:.0e} rel")
    print("=" * 86)

    all_rows = []
    for test in TESTS:
        rows = test()
        all_rows.extend(rows)
        layer = rows[0][0]
        layer_ok = all(r[4] for r in rows)
        print(f"\n  {layer}   {'PASS' if layer_ok else 'FAIL'}")
        for (_, check, kind, metric, ok) in rows:
            print(f"    {'PASS' if ok else 'FAIL'}  {check:<22} {kind:<4} {metric:.2e}")

    n_pass = sum(r[4] for r in all_rows)
    n_total = len(all_rows)
    layers_ok = {}
    for (layer, _, _, _, ok) in all_rows:
        layers_ok[layer] = layers_ok.get(layer, True) and ok
    gate_ok = (n_pass == n_total)

    print("\n" + "=" * 86)
    print(f"  {n_pass}/{n_total} checks pass   |   "
          f"layers passing: {sum(layers_ok.values())}/{len(layers_ok)}")
    print(f"  {'ALL LAYERS MATCH PYTORCH' if gate_ok else 'FAILURES - investigate above'}")

    write_md(all_rows, layers_ok, gate_ok)
    print(f"  wrote {RESULTS_MD}")
    return 0 if gate_ok else 1


def write_md(rows, layers_ok, gate_ok):
    L = []
    L.append("# Layer Parity Results")
    L.append("")
    L.append("Layer-by-layer parity of [`layers.py`](../layers.py) against PyTorch. For each "
             "layer the equivalent PyTorch layer is built with the **same weights**, run "
             "forward + backward on the **same input** with the **same upstream gradient**, "
             "and compared.")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append("- PyTorch is the reference oracle only - not part of the model. It runs in "
             "**float64** so any mismatch is a real bug, not float32 noise.")
    L.append("- The torch reference modules mirror nanoGPT's `model.py`: manual causal "
             "attention, exact-erf GeLU (`F.gelu`), pre-LN block, `nn.LayerNorm` (eps 1e-5, "
             "biased variance).")
    L.append("- `layers.Linear` stores its weight as `(in, out)`; `nn.Linear` stores "
             "`(out, in)` - the harness transposes when copying weights and gradients across.")
    L.append(f"- **Gate:** forward outputs match to `{FWD_ATOL:.0e}` absolute; all parameter "
             f"gradients and the input gradient match to `{GRAD_RTOL:.0e}` relative "
             "(`max|a-b| / max(|a|,|b|)`).")
    L.append("- Embedding has no input gradient (integer ids are not differentiable).")
    L.append("")
    L.append("## Results")
    L.append("")
    L.append("| Layer | Check | Kind | Error | Result |")
    L.append("|---|---|---|---|---|")
    for (layer, check, kind, metric, ok) in rows:
        kind_lbl = "abs" if kind == "abs" else "rel"
        L.append(f"| {layer} | `{check}` | {kind_lbl} | {metric:.2e} | "
                 f"{'PASS' if ok else 'FAIL'} |")
    L.append("")
    n_pass = sum(r[4] for r in rows)
    L.append(f"**{n_pass} / {len(rows)} checks pass. "
             f"{sum(layers_ok.values())} / {len(layers_ok)} layers match PyTorch.**")
    L.append("")
    L.append("## Per-layer verdict")
    L.append("")
    L.append("| Layer | Verdict |")
    L.append("|---|---|")
    for layer, ok in layers_ok.items():
        L.append(f"| {layer} | {'PASS' if ok else 'FAIL'} |")
    L.append("")
    L.append("## How to run")
    L.append("")
    L.append("```")
    L.append(".venv/bin/python tests/test_layer_parity.py")
    L.append("```")
    L.append("")
    if gate_ok:
        L.append("**Every layer matches PyTorch in forward and backward.** The autograd "
                 "`Tensor`, the op library, and the layer assembly are all correct - safe to "
                 "assemble the full GPT model.")
    else:
        L.append("**Some layer did not match** - see the table. Because this is layer-by-layer, "
                 "the failing rows point straight at the broken layer.")
    L.append("")
    with open(RESULTS_MD, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
