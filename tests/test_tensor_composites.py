"""
test_tensor_composites.py - Step 2 gate, composite ops.

The four composite ops - sdpa, multi-head attention, the pre-LN transformer
block, and the full tiny GPT - are rebuilt here by *composing Tensor primitives*.
This is the real test of the autograd graph: the backward of a 30-node graph is
produced entirely by the topological walk + gradient accumulation, with no
hand-written composite backward at all.

Each is compared against the Step 1 hand-rolled composite `backward_fn`:

  - sdpa / mha / transformer_block : BIT-EXACT (zero diff). No tensor is reused
    in a way that changes accumulation order, and every primitive is the same
    closure, so the graph walk reproduces the hand-rolled backward exactly.

  - tiny_gpt : bit-exact on every parameter EXCEPT the weight-tied `W_e`, which
    matches to MACHINE PRECISION (~1 ULP). `W_e` is used twice (token embedding
    + lm_head); Step 1 accumulates the two contributions in place with
    `np.add.at`, the graph accumulates them as `head_grad + embed_grad`. Float
    addition is not associative, so the repeated-token rows can differ in the
    last bit. Same math, both equally correct - it is arithmetic, not a bug.

Run:  .venv/bin/python tests/test_tensor_composites.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))

import ops                       # noqa: E402
import gradcheck as gc           # noqa: E402
import tensor as T               # noqa: E402
from tensor import Tensor        # noqa: E402

MACHINE_EPS_GATE = 1e-12         # "matches to machine precision" threshold


# --------------------------------------------------------------------------
# composite graphs, built purely from Tensor primitives
# --------------------------------------------------------------------------

def _sdpa_graph(Q, K, V):
    """Scaled dot-product causal attention = the 4 verified primitives chained."""
    S = T.attention_scores(Q, K)
    P = T.softmax(T.causal_mask(S))            # default fill -1e9, matches ops.sdpa
    return T.attention_output(P, V)


def _mha_graph(x, W_attn, b_attn, W_proj, b_proj, H):
    """Multi-head causal self-attention, composed from primitives."""
    B, Tn, D = x.shape
    d = D // H
    qkv = T.linear(x, W_attn, b_attn)                       # (B,T,3D)
    Qf, Kf, Vf = T.split3(qkv)                              # each (B,T,D)

    def heads(z):                                           # (B,T,D) -> (B,H,T,d)
        return T.permute_heads(T.reshape(z, (B, Tn, H, d)))

    O = _sdpa_graph(heads(Qf), heads(Kf), heads(Vf))        # (B,H,T,d)
    O_merged = T.reshape(T.permute_heads(O), (B, Tn, D))    # (B,T,D)
    return T.linear(O_merged, W_proj, b_proj)


def _block_graph(x, p, H, eps):
    """Pre-LN transformer block. `p` maps the 12 block-param names to Tensors.
    Note `r` is used twice (LN2 input and the residual-2 skip) - the autograd
    graph accumulates both contributions, and in the same order Step 1 did."""
    a = T.layernorm(x, p['ln1_g'], p['ln1_b'], eps)
    u = _mha_graph(a, p['W_attn'], p['b_attn'], p['W_proj'], p['b_proj'], H)
    r = x + u                                               # residual 1
    c = T.layernorm(r, p['ln2_g'], p['ln2_b'], eps)
    h1 = T.linear(c, p['W1'], p['mlp_b1'])
    h2 = T.gelu(h1)
    m = T.linear(h2, p['W2'], p['mlp_b2'])
    return r + m                                            # residual 2


def _tiny_gpt_graph(t, i):
    """Full GPT-2 forward. `W_e` is passed to BOTH the token embedding and the
    lm_head - one Tensor, two consumers - so the graph accumulates the tied
    gradient automatically."""
    H, n_blocks, eps = i['H'], i['n_blocks'], i['eps']
    ids, targets = i['ids'], i['targets']
    B, Tn = ids.shape
    x = T.embedding(t['W_e'], ids) + T.positional_embedding(t['W_p'], Tn, B)
    for bi in range(n_blocks):
        p = {pn: t[f'b{bi}_{pn}'] for pn in ops.BLOCK_PARAMS}
        x = _block_graph(x, p, H, eps)
    xf = T.layernorm(x, t['lnf_g'], t['lnf_b'], eps)
    logits = T.lm_head(xf, t['W_e'])                        # weight tying
    return T.softmax_cross_entropy(logits, targets)


WIRING = {
    "scaled dot-product attention (causal)":
        lambda t, i: _sdpa_graph(t['Q'], t['K'], t['V']),
    "multi-head attention (full module)":
        lambda t, i: _mha_graph(t['x'], t['W_attn'], t['b_attn'],
                                t['W_proj'], t['b_proj'], i['H']),
    "transformer block (pre-LN)":
        lambda t, i: _block_graph(t['x'], {k: t[k] for k in ops.BLOCK_PARAMS},
                                  i['H'], i['eps']),
    "tiny GPT (full model, weight-tied)":
        _tiny_gpt_graph,
}


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

def run_one(name, seed, builder):
    np.random.seed(seed)
    ops_fn, inputs, diff_names = builder()

    # Step 1 reference
    out_ref, bwd_ref = ops_fn(**inputs)
    out_ref = np.asarray(out_ref, dtype=np.float64)
    if out_ref.ndim == 0:
        upstream = np.asarray(np.random.randn(), dtype=np.float64)
    else:
        upstream = np.random.randn(*out_ref.shape).astype(np.float64)
    ref_grads = bwd_ref(upstream)

    # Tensor graph
    tdict = {k: Tensor(inputs[k], requires_grad=True) for k in diff_names}
    out_t = WIRING[name](tdict, inputs)
    fwd_exact = np.array_equal(out_t.data, out_ref)
    L = (out_t * Tensor(upstream)).sum()
    L.backward()

    rows = []
    for k in diff_names:
        got, want = tdict[k].grad, ref_grads[k]
        if got is None or got.shape != want.shape:
            rows.append((k, tuple(np.shape(want)), float('inf'), False, "shape/none"))
            continue
        diff = float(np.abs(got - want).max())
        if np.array_equal(got, want):
            verdict = "bit-exact"
        elif diff < MACHINE_EPS_GATE:
            verdict = "machine-eps"
        else:
            verdict = "FAIL"
        rows.append((k, tuple(want.shape), diff, fwd_exact, verdict))
    return rows, fwd_exact


def main():
    print("Step 2 gate - composite Tensor graphs vs Step 1 hand-rolled backward")
    print("=" * 96)
    composites = [(n, it, s, b) for (n, it, s, b) in gc.TESTS if n in WIRING]

    n_bitexact = n_macheps = n_fail = 0
    for name, item, seed, builder in composites:
        rows, fwd_exact = run_one(name, seed, builder)
        print(f"\n  {name}   (item {item})   forward bit-exact: {fwd_exact}")
        for (inp, shape, diff, fwd_ok, verdict) in rows:
            if verdict == "bit-exact":
                n_bitexact += 1
            elif verdict == "machine-eps":
                n_macheps += 1
            else:
                n_fail += 1
            tag = {"bit-exact": "PASS", "machine-eps": "PASS", }.get(verdict, "FAIL")
            print(f"    {tag}  d{inp:<14} shape {str(shape):<16} "
                  f"max|diff|={diff:.2e}  [{verdict}]")

    n_total = n_bitexact + n_macheps + n_fail
    print("\n" + "=" * 96)
    print(f"  {n_bitexact}/{n_total} bit-exact (zero diff)   |   "
          f"{n_macheps}/{n_total} machine-precision (< {MACHINE_EPS_GATE:.0e}, weight-tied W_e)   |   "
          f"{n_fail} fail")
    ok = (n_fail == 0)
    print(f"  {'ALL CLEAR - autograd graph reproduces Step 1' if ok else 'FAILURES - investigate above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
