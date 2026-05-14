"""
test_tensor.py - Step 2 gate.

Rebuild every Step 1 op end-to-end with the autograd `Tensor` graph, call
`.backward()`, and confirm `x.grad` is BIT-IDENTICAL to the Step 1 hand-rolled
`backward_fn` result - `np.array_equal`, zero diff, not "within tolerance".

Why bit-exact is the right bar: every primitive Tensor op wraps the *exact*
ops.py closure that tests/gradcheck.py finite-difference verified. So the Tensor
graph and the Step 1 reference are literally the same code path - the only new
machinery between them is the topological walk and gradient accumulation, and
neither changes a single floating-point value when a tensor is used once.

For each op the harness:
  1. builds the same inputs Step 1 used (reusing gradcheck's builders + seeds)
  2. runs the Step 1 reference: out, bwd = ops.<op>(...); ref = bwd(upstream)
  3. runs the Tensor graph: L = (out_tensor * upstream).sum(); L.backward()
  4. asserts x.grad == ref bit-for-bit

Composites (sdpa / mha / transformer_block / tiny_gpt) are rebuilt by *composing
Tensor primitives* in test_tensor_composites.py - that is the real autograd-graph
test, and matches Step 1 to machine precision.

Run:  .venv/bin/python tests/test_tensor.py
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # tests/  -> ops, gradcheck
sys.path.insert(0, os.path.dirname(_HERE))      # repo root -> tensor

import ops                       # noqa: E402
import gradcheck as gc           # noqa: E402  (reused only for builders + seeds)
import tensor as T               # noqa: E402
from tensor import Tensor        # noqa: E402

# Composite ops are tested in test_tensor_composites.py (composed from primitives).
COMPOSITES = {
    "scaled dot-product attention (causal)",
    "multi-head attention (full module)",
    "transformer block (pre-LN)",
    "tiny GPT (full model, weight-tied)",
}

# How to wire each Step 1 op with Tensor operations.
# wire(t, i): t = {name: Tensor} differentiable inputs, i = raw inputs dict (consts).
WIRING = {
    "add":                                  lambda t, i: t['a'] + t['b'],
    "add (bias broadcast)":                 lambda t, i: t['x'] + t['b'],
    "mul (elementwise)":                    lambda t, i: t['a'] * t['b'],
    "add (broadcasting)":                   lambda t, i: t['a'] + t['b'],
    "mul (broadcasting)":                   lambda t, i: t['a'] * t['b'],
    "scale":                                lambda t, i: T.scale(t['x'], i['c']),
    "relu":                                 lambda t, i: T.relu(t['x']),
    "gelu (exact erf)":                     lambda t, i: T.gelu(t['x']),
    "matmul":                               lambda t, i: t['x'] @ t['W'],
    "matmul (batched x)":                   lambda t, i: t['x'] @ t['W'],
    "linear (affine)":                      lambda t, i: T.linear(t['x'], t['W'], t['b']),
    "sum reduction":                        lambda t, i: T.sum_lastaxis(t['x']),
    "mean reduction":                       lambda t, i: T.mean_lastaxis(t['x']),
    "sum (all elements)":                   lambda t, i: t['x'].sum(),
    "transpose (2D)":                       lambda t, i: T.transpose(t['x']),
    "permute (0,2,1,3)":                    lambda t, i: T.permute_heads(t['x']),
    "reshape":                              lambda t, i: T.reshape(t['x'], i['newshape']),
    "slice (last axis)":                    lambda t, i: T.slice_lastaxis(t['x'], i['start'], i['stop']),
    "softmax":                              lambda t, i: T.softmax(t['x']),
    "softmax (key dim, 4D)":                lambda t, i: T.softmax(t['x']),
    "cross-entropy (from probs)":           lambda t, i: T.cross_entropy(t['p'], i['y']),
    "softmax + CE fused":                   lambda t, i: T.softmax_cross_entropy(t['logits'], i['targets']),
    "layernorm":                            lambda t, i: T.layernorm(t['x'], t['gamma'], t['beta'], i['eps']),
    "embedding (np.add.at)":                lambda t, i: T.embedding(t['W_e'], i['ids']),
    "positional embedding":                 lambda t, i: T.positional_embedding(t['W_p'], i['T'], i['B']),
    "lm_head (weight-tied proj)":           lambda t, i: T.lm_head(t['h'], t['W_e']),
    "attention scores  S = QK^T/sqrt(d)":   lambda t, i: T.attention_scores(t['Q'], t['K']),
    "causal mask":                          lambda t, i: T.causal_mask(t['S'], i['fill']),
    "attention output  O = PV":             lambda t, i: T.attention_output(t['P'], t['V']),
}


def run_one(name, seed, builder):
    """Return list of (input_name, shape, max_abs_diff, bit_exact) rows."""
    np.random.seed(seed)
    ops_fn, inputs, diff_names = builder()

    # --- Step 1 reference: the verified hand-rolled backward ---
    out_ref, bwd_ref = ops_fn(**inputs)
    out_ref = np.asarray(out_ref, dtype=np.float64)
    if out_ref.ndim == 0:
        upstream = np.asarray(np.random.randn(), dtype=np.float64)
    else:
        upstream = np.random.randn(*out_ref.shape).astype(np.float64)
    ref_grads = bwd_ref(upstream)

    # --- Tensor graph: same inputs, same upstream, end-to-end autograd ---
    tdict = {k: Tensor(inputs[k], requires_grad=True) for k in diff_names}
    out_t = WIRING[name](tdict, inputs)
    # forward must agree first (else backward comparison is meaningless)
    fwd_exact = np.array_equal(out_t.data, out_ref)
    L = (out_t * Tensor(upstream)).sum()
    L.backward()

    rows = []
    for k in diff_names:
        got, want = tdict[k].grad, ref_grads[k]
        if got is None:
            rows.append((k, tuple(np.shape(want)), float('inf'), False, fwd_exact))
            continue
        exact = (got.shape == want.shape) and np.array_equal(got, want)
        md = float(np.abs(got - want).max()) if got.shape == want.shape else float('inf')
        rows.append((k, tuple(want.shape), md, exact, fwd_exact))
    return rows


def main():
    print("Step 2 gate - Tensor autograd vs Step 1 hand-rolled backward (bit-exact)")
    print("=" * 94)
    primitives = [(n, it, s, b) for (n, it, s, b) in gc.TESTS if n not in COMPOSITES]

    all_rows, missing = [], []
    for name, item, seed, builder in primitives:
        if name not in WIRING:
            missing.append(name)
            print(f"  MISSING WIRING  {name}")
            continue
        rows = run_one(name, seed, builder)
        for (inp, shape, md, exact, fwd_exact) in rows:
            all_rows.append((name, item, inp, shape, md, exact, fwd_exact))
            tag = "PASS" if (exact and fwd_exact) else "FAIL"
            label = name if len(rows) == 1 else f"{name}  [d{inp}]"
            note = "" if fwd_exact else "  <- FORWARD MISMATCH"
            print(f"  {tag}  {label:<54} item {item:<6} shape {str(shape):<15} "
                  f"max|diff|={md:.1e}{note}")

    n_exact = sum(1 for r in all_rows if r[5] and r[6])
    n_total = len(all_rows)
    print("=" * 94)
    print(f"  {n_exact}/{n_total} primitive gradient checks are BIT-EXACT vs Step 1   "
          f"({len(COMPOSITES)} composites -> test_tensor_composites.py)")
    ok = (n_exact == n_total) and not missing
    print(f"  {'ALL BIT-EXACT' if ok else 'NOT ALL EXACT - investigate above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
