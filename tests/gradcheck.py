"""
gradcheck.py - Finite-difference verification of every gradient formula in
../derivation.md, via the standalone ops in ops.py.

Method (per op):
  1. build small random float64 inputs
  2. out, backward_fn = op(**inputs)
  3. draw a random upstream gradient; loss L = sum(out * upstream)
  4. analytic_grad = backward_fn(upstream)
  5. numeric_grad via central finite difference:
         L_plus  = sum(op(x with x[i]+eps)[0] * upstream)
         L_minus = sum(op(x with x[i]-eps)[0] * upstream)
         numeric[i] = (L_plus - L_minus) / (2*eps)
  6. compare analytic vs numeric

Settings: eps = 1e-6, float64, central difference.

Pass criterion (per element, the standard combined criterion used by
np.allclose and torch.autograd.gradcheck):

    |a - n| < ATOL    OR    |a - n| / max(|a|,|n|) < RTOL

RTOL = 1e-5 is the relative gate the task specifies. ATOL = 1e-7 is required
because some gradients are *exactly zero* (e.g. the attention key-bias, by
softmax shift-invariance) or sit below the finite-difference noise floor
(~2e-8 for these ops) - and there a relative error is just rounding-noise
divided by rounding-noise. A genuinely wrong formula misses by many orders of
magnitude more (see the negative control: the buggy embedding misses by 2.49).

Run:    .venv/bin/python tests/gradcheck.py
Writes: tests/gradcheck_results.md   (exit 0 iff every check passes)
"""
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ops  # noqa: E402

EPS = 1e-6            # finite-difference step (task-specified)
RTOL = 1e-5           # relative tolerance - the gate the task specifies
ATOL = 1e-7           # absolute tolerance - for gradients that are genuinely
                      # zero or below the FD noise floor (~2e-8 for these ops)
_REL_FLOOR = 1e-4     # report max_rel_error only over gradient entries above
                      # this magnitude (smaller entries are governed by ATOL)
RESULTS_MD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gradcheck_results.md")


# --------------------------------------------------------------------------
# harness
# --------------------------------------------------------------------------

def _compare(analytic, numeric):
    """Return (max_abs_error, max_rel_error, passed).

    `passed` uses the combined criterion: every element must agree either
    absolutely (|a-n| < ATOL) or relatively (|a-n| < RTOL*max(|a|,|n|)).
    `max_abs_error` is over all entries. `max_rel_error` is over entries whose
    gradient magnitude exceeds `_REL_FLOOR` - the entries large enough that a
    relative error is the meaningful metric.
    """
    a = np.asarray(analytic, dtype=np.float64)
    n = np.asarray(numeric, dtype=np.float64)
    diff = np.abs(a - n)
    denom = np.maximum(np.abs(a), np.abs(n))

    elem_ok = (diff < ATOL) | (diff < RTOL * denom)
    passed = bool(np.all(elem_ok))

    max_abs = float(diff.max())
    big = denom > _REL_FLOOR
    max_rel = float((diff[big] / denom[big]).max()) if np.any(big) else 0.0
    return max_abs, max_rel, passed


def gradcheck_op(name, item, op, inputs, diff_names, eps=EPS):
    """Finite-difference check `op` w.r.t. every input named in `diff_names`."""
    out, backward_fn = op(**inputs)
    out = np.asarray(out, dtype=np.float64)
    if out.ndim == 0:
        upstream = np.asarray(np.random.randn(), dtype=np.float64)
    else:
        upstream = np.random.randn(*out.shape).astype(np.float64)

    analytic_grads = backward_fn(upstream)

    rows = []
    for nm in diff_names:
        x = inputs[nm]
        if not isinstance(x, np.ndarray) or x.dtype != np.float64:
            raise TypeError(f"{name}/{nm}: differentiable input must be a float64 array")
        analytic = np.asarray(analytic_grads[nm], dtype=np.float64)
        if analytic.shape != x.shape:
            raise ValueError(
                f"{name}/{nm}: analytic grad shape {analytic.shape} != input shape {x.shape}")

        numeric = np.zeros_like(x)
        for idx in np.ndindex(*x.shape):
            orig = x[idx]
            x[idx] = orig + eps
            L_plus = float(np.sum(np.asarray(op(**inputs)[0], dtype=np.float64) * upstream))
            x[idx] = orig - eps
            L_minus = float(np.sum(np.asarray(op(**inputs)[0], dtype=np.float64) * upstream))
            x[idx] = orig
            numeric[idx] = (L_plus - L_minus) / (2.0 * eps)

        abs_err, rel_err, passed = _compare(analytic, numeric)
        rows.append(dict(name=name, item=item, input=nm, shape=tuple(x.shape),
                         abs_err=abs_err, rel_err=rel_err, passed=passed))
    return rows


# --------------------------------------------------------------------------
# input builders  (caller seeds the RNG before invoking each one)
# --------------------------------------------------------------------------

def b_add():
    return ops.add, {'a': np.random.randn(3, 4), 'b': np.random.randn(3, 4)}, ['a', 'b']

def b_add_bias():
    return ops.add_bias, {'x': np.random.randn(3, 4), 'b': np.random.randn(4)}, ['x', 'b']

def b_mul():
    return ops.mul, {'a': np.random.randn(3, 4), 'b': np.random.randn(3, 4)}, ['a', 'b']

def b_add_bc():
    return ops.add, {'a': np.random.randn(3, 4), 'b': np.random.randn(4)}, ['a', 'b']

def b_mul_bc():
    return ops.mul, {'a': np.random.randn(2, 3, 4), 'b': np.random.randn(3, 1)}, ['a', 'b']

def b_scale():
    return ops.scale, {'x': np.random.randn(3, 4), 'c': 1.0 / np.sqrt(8.0)}, ['x']

def b_relu():
    x = np.random.randn(3, 4)
    x[np.abs(x) < 1e-3] += 0.5            # keep well clear of the kink at 0
    return ops.relu, {'x': x}, ['x']

def b_gelu():
    return ops.gelu, {'x': np.random.randn(3, 4)}, ['x']

def b_matmul():
    return ops.matmul, {'x': np.random.randn(3, 4), 'W': np.random.randn(4, 5)}, ['x', 'W']

def b_matmul_batched():
    # 3-D x: exercises the flattened dW contraction (not the 2-D X^T dY shortcut)
    return ops.matmul, {'x': np.random.randn(2, 3, 4), 'W': np.random.randn(4, 5)}, ['x', 'W']

def b_linear():
    return ops.linear, {'x': np.random.randn(3, 4), 'W': np.random.randn(4, 5),
                        'b': np.random.randn(5)}, ['x', 'W', 'b']

def b_sum():
    return ops.sum_lastaxis, {'x': np.random.randn(3, 4)}, ['x']

def b_mean():
    return ops.mean_lastaxis, {'x': np.random.randn(3, 4)}, ['x']

def b_sum_all():
    return ops.sum_all, {'x': np.random.randn(3, 4)}, ['x']

def b_transpose2d():
    return ops.transpose2d, {'x': np.random.randn(3, 4)}, ['x']

def b_permute():
    return ops.permute_bhtd, {'x': np.random.randn(2, 3, 4, 5)}, ['x']

def b_reshape():
    return ops.reshape_op, {'x': np.random.randn(3, 4), 'newshape': (2, 6)}, ['x']

def b_slice():
    return ops.slice_lastaxis, {'x': np.random.randn(2, 3, 12), 'start': 4, 'stop': 8}, ['x']

def b_softmax():
    return ops.softmax_op, {'x': np.random.randn(3, 4)}, ['x']

def b_softmax_4d():
    return ops.softmax_op, {'x': np.random.randn(1, 2, 4, 4)}, ['x']

def b_ce_probs():
    logits = np.random.randn(3, 5)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    p = e / e.sum(axis=-1, keepdims=True)            # valid probability rows
    y = np.zeros((3, 5))
    y[np.arange(3), np.random.randint(0, 5, 3)] = 1.0
    return ops.cross_entropy_from_probs, {'p': p, 'y': y}, ['p']

def b_softmax_ce():
    return ops.softmax_ce_fused, {'logits': np.random.randn(2, 3, 5),
                                  'targets': np.random.randint(0, 5, size=(2, 3))}, ['logits']

def b_layernorm():
    return ops.layernorm, {'x': np.random.randn(3, 4), 'gamma': np.random.randn(4),
                           'beta': np.random.randn(4), 'eps': 1e-5}, ['x', 'gamma', 'beta']

def b_embedding():
    # ids hand-picked so token 0 appears 3x and token 2 appears 2x: the
    # repeated-id case that np.add.at handles and `+=` silently breaks.
    return ops.embedding, {'W_e': np.random.randn(6, 4),
                           'ids': np.array([[0, 2, 0], [2, 1, 0]])}, ['W_e']

def b_embedding_buggy():
    return ops.embedding_buggy, {'W_e': np.random.randn(6, 4),
                                 'ids': np.array([[0, 2, 0], [2, 1, 0]])}, ['W_e']

def b_pos_embedding():
    return ops.positional_embedding, {'W_p': np.random.randn(5, 4), 'T': 3, 'B': 2}, ['W_p']

def b_lm_head():
    return ops.lm_head, {'h': np.random.randn(2, 3, 4), 'W_e': np.random.randn(6, 4)}, ['h', 'W_e']

def b_attn_scores():
    return ops.attention_scores, {'Q': np.random.randn(1, 2, 4, 8),
                                  'K': np.random.randn(1, 2, 4, 8)}, ['Q', 'K']

def b_causal_mask():
    # moderate fill keeps the FD subtraction well-conditioned; the backward
    # (dS = dS_masked) is value-independent. The realistic -1e9 fill is
    # exercised - cleanly, via exp underflow to exactly 0 - inside sdpa / mha.
    return ops.causal_mask, {'S': np.random.randn(1, 2, 4, 4), 'fill': -30.0}, ['S']

def b_attn_output():
    return ops.attention_output, {'P': np.random.randn(1, 2, 4, 4),
                                  'V': np.random.randn(1, 2, 4, 8)}, ['P', 'V']

def b_sdpa():
    # the shape the task asks for: T=4, H=2, d=8 (B=2)
    B, H, T, d = 2, 2, 4, 8
    return ops.sdpa_causal, {'Q': np.random.randn(B, H, T, d),
                             'K': np.random.randn(B, H, T, d),
                             'V': np.random.randn(B, H, T, d)}, ['Q', 'K', 'V']

def _winit(fan_in, fan_out):
    """randn / sqrt(fan_in): the standard variance-preserving init. It keeps
    activations ~unit scale on the forward pass AND keeps the gradient norm
    ~preserved on the backward pass, so the finite-difference check stays
    well-conditioned even through a deep composition. With a strongly
    attenuating init instead (e.g. randn * 0.1), the deepest-parameter
    gradients shrink below the FD noise floor (~|L| * 1e-10) and the *check*
    fails even though the *formula* is correct - see gradcheck_results.md."""
    return np.random.randn(fan_in, fan_out) / np.sqrt(fan_in)

def _block_params(D, H):
    F = 4 * D
    return {
        'ln1_g': np.random.randn(D), 'ln1_b': np.random.randn(D),
        'W_attn': _winit(D, 3 * D), 'b_attn': np.random.randn(3 * D),
        'W_proj': _winit(D, D), 'b_proj': np.random.randn(D),
        'ln2_g': np.random.randn(D), 'ln2_b': np.random.randn(D),
        'W1': _winit(D, F), 'mlp_b1': np.random.randn(F),
        'W2': _winit(F, D), 'mlp_b2': np.random.randn(D),
    }

def b_mha():
    B, T, H, d = 2, 4, 2, 8
    D = H * d
    return ops.mha, {'x': np.random.randn(B, T, D),
                     'W_attn': _winit(D, 3 * D), 'b_attn': np.random.randn(3 * D),
                     'W_proj': _winit(D, D), 'b_proj': np.random.randn(D),
                     'H': H}, \
        ['x', 'W_attn', 'b_attn', 'W_proj', 'b_proj']

def b_block():
    B, T, H, d = 2, 4, 2, 4
    D = H * d
    inputs = {'x': np.random.randn(B, T, D), 'H': H, 'eps': 1e-5}
    inputs.update(_block_params(D, H))
    return ops.transformer_block, inputs, ['x'] + ops.BLOCK_PARAMS

def b_tiny_gpt():
    B, T, V, D, H, n_blocks, Tmax = 2, 3, 7, 8, 2, 2, 4
    inputs = {
        'ids': np.array([[1, 3, 1], [0, 1, 5]]),        # token 1 repeats
        'targets': np.array([[3, 1, 0], [1, 5, 2]]),
        'W_e': np.random.randn(V, D) / np.sqrt(D),
        'W_p': np.random.randn(Tmax, D) / np.sqrt(D),
        'lnf_g': np.random.randn(D), 'lnf_b': np.random.randn(D),
        'H': H, 'n_blocks': n_blocks, 'eps': 1e-5,
    }
    diff = ['W_e', 'W_p', 'lnf_g', 'lnf_b']
    for i in range(n_blocks):
        for pn, val in _block_params(D, H).items():
            inputs[f'b{i}_{pn}'] = val
            diff.append(f'b{i}_{pn}')
    return ops.tiny_gpt, inputs, diff


# (display name, derivation item, seed, builder)
TESTS = [
    ("add", "2", 101, b_add),
    ("add (bias broadcast)", "2", 102, b_add_bias),
    ("mul (elementwise)", "3", 128, b_mul),
    ("add (broadcasting)", "2", 129, b_add_bc),
    ("mul (broadcasting)", "3", 130, b_mul_bc),
    ("scale", "3", 103, b_scale),
    ("relu", "4a", 104, b_relu),
    ("gelu (exact erf)", "4b", 105, b_gelu),
    ("matmul", "5", 106, b_matmul),
    ("matmul (batched x)", "5", 133, b_matmul_batched),
    ("linear (affine)", "5", 107, b_linear),
    ("sum reduction", "6", 108, b_sum),
    ("mean reduction", "6", 109, b_mean),
    ("sum (all elements)", "6", 131, b_sum_all),
    ("transpose (2D)", "7", 110, b_transpose2d),
    ("permute (0,2,1,3)", "7", 111, b_permute),
    ("reshape", "7", 112, b_reshape),
    ("slice (last axis)", "7", 132, b_slice),
    ("softmax", "8", 113, b_softmax),
    ("softmax (key dim, 4D)", "18", 114, b_softmax_4d),
    ("cross-entropy (from probs)", "9", 115, b_ce_probs),
    ("softmax + CE fused", "10", 116, b_softmax_ce),
    ("layernorm", "11", 117, b_layernorm),
    ("embedding (np.add.at)", "12", 118, b_embedding),
    ("positional embedding", "13", 119, b_pos_embedding),
    ("lm_head (weight-tied proj)", "14", 120, b_lm_head),
    ("attention scores  S = QK^T/sqrt(d)", "16", 121, b_attn_scores),
    ("causal mask", "17", 122, b_causal_mask),
    ("attention output  O = PV", "19", 123, b_attn_output),
    ("scaled dot-product attention (causal)", "20", 124, b_sdpa),
    ("multi-head attention (full module)", "15+20+21", 125, b_mha),
    ("transformer block (pre-LN)", "22", 126, b_block),
    ("tiny GPT (full model, weight-tied)", "23", 127, b_tiny_gpt),
]
NEG_CONTROL = ("embedding (BUGGY  dW[ids] += dY)", "12", 118, b_embedding_buggy)


# --------------------------------------------------------------------------
# special checks (structural / consistency, not finite-difference)
# --------------------------------------------------------------------------

def special_fused_vs_composed():
    """Item 10: the fused softmax+CE gradient must equal softmax then CE composed."""
    np.random.seed(777)
    logits = np.random.randn(2, 3, 5)
    targets = np.random.randint(0, 5, size=(2, 3))
    B, T, V = logits.shape
    N = B * T

    L_fused, bw_fused = ops.softmax_ce_fused(logits, targets)
    up = np.asarray(np.random.randn(), dtype=np.float64)
    d_fused = bw_fused(up)['logits']

    p, bw_sm = ops.softmax_op(logits)
    onehot = np.zeros_like(p)
    onehot[np.arange(B)[:, None], np.arange(T)[None, :], targets] = 1.0
    L_comp = -np.sum(onehot * np.log(p)) / N
    d_comp = bw_sm((-onehot / p / N) * up)['x']

    return abs(float(L_fused) - float(L_comp)), float(np.abs(d_fused - d_comp).max())


def special_kbias_zero():
    """The K-part of the fused QKV bias has a gradient that is *exactly zero*:
    adding a constant to every key shifts every score in a query's softmax row
    by the same amount, and softmax is shift-invariant - so the loss does not
    depend on the key bias at all. This is why a pure relative-error check on
    `db_attn` looks like a failure (it divides FD noise by FD noise); the
    formula is correct, the gradient is genuinely zero. Confirms: |db_K| ~ 0
    while |db_Q| and |db_V| are not."""
    np.random.seed(909)
    B, T, H, d = 2, 4, 2, 8
    D = H * d
    x = np.random.randn(B, T, D)
    W_attn = np.random.randn(D, 3 * D) / np.sqrt(D)
    b_attn = np.random.randn(3 * D)
    W_proj = np.random.randn(D, D) / np.sqrt(D)
    b_proj = np.random.randn(D)
    Y, bw = ops.mha(x, W_attn, b_attn, W_proj, b_proj, H)
    db = bw(np.random.randn(*Y.shape))['b_attn']
    q_mag = float(np.abs(db[:D]).max())
    k_mag = float(np.abs(db[D:2 * D]).max())          # <- expected ~0
    v_mag = float(np.abs(db[2 * D:]).max())
    return q_mag, k_mag, v_mag


def special_adamw():
    """Item 24: reproduce the worked example, and confirm AdamW(wd=0) == Adam."""
    def adamw_step(theta, g, m, v, t, lr, b1, b2, eps, wd):
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1 ** t)
        vhat = v / (1 - b2 ** t)
        theta = theta - lr * mhat / (np.sqrt(vhat) + eps) - lr * wd * theta
        return theta, m, v

    theta, _, _ = adamw_step(1.0, 0.1, 0.0, 0.0, 1, 0.1, 0.9, 0.95, 1e-8, 0.1)
    err_worked = abs(theta - 0.89000001)            # value from derivation.md item 24

    theta_wd0, _, _ = adamw_step(1.0, 0.1, 0.0, 0.0, 1, 0.1, 0.9, 0.95, 1e-8, 0.0)
    m = 0.1 * 0.1
    v = 0.05 * 0.01
    mhat, vhat = m / (1 - 0.9), v / (1 - 0.95)
    theta_adam = 1.0 - 0.1 * mhat / (np.sqrt(vhat) + 1e-8)
    err_decoupling = abs(theta_wd0 - theta_adam)
    return err_worked, err_decoupling


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------

def main():
    print(f"gradient check  |  eps={EPS:.0e}  float64  central difference")
    print(f"pass criterion (per element):  |a-n| < {ATOL:.0e}   OR   |a-n|/max(|a|,|n|) < {RTOL:.0e}")
    print("=" * 92)

    all_rows = []
    errored = []
    t0 = time.time()

    for name, item, seed, builder in TESTS:
        np.random.seed(seed)
        try:
            op, inputs, diff_names = builder()
            rows = gradcheck_op(name, item, op, inputs, diff_names)
        except Exception as exc:                                  # noqa: BLE001
            errored.append((name, exc))
            print(f"  ERROR  {name}: {exc}")
            traceback.print_exc()
            continue
        all_rows.extend(rows)
        for r in rows:
            tag = "PASS" if r['passed'] else "FAIL"
            label = r['name'] if len(rows) == 1 else f"{r['name']}  [d{r['input']}]"
            print(f"  {tag}  {label:<52} item {r['item']:<9} "
                  f"abs={r['abs_err']:.2e}  rel={r['rel_err']:.2e}")

    # negative control: the buggy embedding MUST fail
    nc_name, nc_item, nc_seed, nc_builder = NEG_CONTROL
    np.random.seed(nc_seed)
    op, inputs, diff_names = nc_builder()
    nc_rows = gradcheck_op(nc_name, nc_item, op, inputs, diff_names)
    nc_failed_as_expected = all(not r['passed'] for r in nc_rows)
    nc_row = nc_rows[0]
    print("-" * 92)
    print(f"  {'FAIL (expected)' if nc_failed_as_expected else 'UNEXPECTED PASS'}  "
          f"{nc_name:<52} item {nc_item:<9} abs={nc_row['abs_err']:.2e}  rel={nc_row['rel_err']:.2e}")

    # special checks
    fwd_diff, grad_diff = special_fused_vs_composed()
    fused_ok = (fwd_diff < 1e-10) and (grad_diff < 1e-10)
    q_mag, k_mag, v_mag = special_kbias_zero()
    kbias_ok = (k_mag < 1e-7) and (q_mag > 1e-3) and (v_mag > 1e-3)
    worked_err, decouple_err = special_adamw()
    adamw_ok = (worked_err < 1e-7) and (decouple_err < 1e-15)
    print("-" * 92)
    print(f"  {'PASS' if fused_ok else 'FAIL'}  softmax+CE fused == composed   "
          f"fwd_diff={fwd_diff:.2e}  grad_diff={grad_diff:.2e}")
    print(f"  {'PASS' if kbias_ok else 'FAIL'}  attention key-bias gradient is exactly 0   "
          f"|dQ|={q_mag:.2e}  |dK|={k_mag:.2e}  |dV|={v_mag:.2e}")
    print(f"  {'PASS' if adamw_ok else 'FAIL'}  AdamW worked example + decoupling   "
          f"worked_err={worked_err:.2e}  decouple_err={decouple_err:.2e}")

    elapsed = time.time() - t0
    n_pass = sum(r['passed'] for r in all_rows)
    n_total = len(all_rows)
    specials_ok = fused_ok and kbias_ok and adamw_ok
    gate_ok = (n_pass == n_total) and not errored and nc_failed_as_expected and specials_ok

    print("=" * 92)
    print(f"  {n_pass}/{n_total} gradient checks pass   |   "
          f"negative control fails as expected: {nc_failed_as_expected}   |   "
          f"special checks: {'ok' if specials_ok else 'FAILED'}   |   {elapsed:.1f}s")
    print(f"  {'ALL CLEAR' if gate_ok else 'NOT CLEAR - investigate above'}")

    write_md(all_rows, (nc_name, nc_item, nc_row, nc_failed_as_expected),
             (fwd_diff, grad_diff, fused_ok), (q_mag, k_mag, v_mag, kbias_ok),
             (worked_err, decouple_err, adamw_ok), elapsed, gate_ok)
    print(f"  wrote {RESULTS_MD}")
    return 0 if gate_ok else 1


def write_md(rows, neg_control, fused, kbias, adamw, elapsed, gate_ok):
    nc_name, nc_item, nc_row, nc_ok = neg_control
    fwd_diff, grad_diff, fused_ok = fused
    q_mag, k_mag, v_mag, kbias_ok = kbias
    worked_err, decouple_err, adamw_ok = adamw
    n_pass = sum(r['passed'] for r in rows)
    n_total = len(rows)

    L = []
    L.append("# Gradient Check Results")
    L.append("")
    L.append("Finite-difference verification of every gradient formula derived in "
             "[`derivation.md`](../derivation.md), implemented as standalone NumPy "
             "`(output, backward_fn)` ops in [`ops.py`](ops.py).")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append("For each op: build small random `float64` inputs, run the forward, draw a random "
             "upstream gradient, and form the scalar `L = sum(output * upstream)`. The analytic "
             "gradient comes from the op's `backward_fn`; the numeric gradient comes from a "
             "central finite difference of `L` w.r.t. each input element:")
    L.append("")
    L.append("```")
    L.append("numeric[i] = ( L(x[i]+eps) - L(x[i]-eps) ) / (2*eps)")
    L.append("```")
    L.append("")
    L.append(f"- **eps** = `{EPS:.0e}`, all arithmetic in **float64**, central difference.")
    L.append(f"- **Pass criterion (per element):** `|a - n| < {ATOL:.0e}` (absolute) **OR** "
             f"`|a - n| / max(|a|,|n|) < {RTOL:.0e}` (relative). This is the standard combined "
             "criterion (`np.allclose`, `torch.autograd.gradcheck`). The relative tolerance "
             f"`{RTOL:.0e}` is the task's gate; the absolute tolerance `{ATOL:.0e}` is required "
             "because some gradients are *exactly zero* (e.g. the attention key-bias - see "
             "special checks) or sit below the finite-difference noise floor (empirically "
             "`~2e-8` for these ops), where a *relative* error is rounding-noise divided by "
             "rounding-noise. A genuinely wrong formula misses by orders of magnitude more - "
             "see the negative control.")
    L.append("- `max_abs_error` = `max |analytic - numeric|` over all entries.")
    L.append(f"- `max_rel_error` = `max |a - n| / max(|a|,|n|)` over entries with gradient "
             f"magnitude `> {_REL_FLOOR:.0e}` (the entries where a relative error is the "
             "meaningful metric; smaller entries are governed by `max_abs_error`).")
    L.append(f"- Reproducible: each op uses a fixed RNG seed. Total runtime {elapsed:.1f}s.")
    L.append("")
    L.append("## Results")
    L.append("")
    L.append("| Op | Derivation item | Input checked | Shape | Max abs error | Max rel error | Result |")
    L.append("|----|----|----|----|----|----|----|")
    for r in rows:
        L.append(f"| {r['name']} | {r['item']} | `{r['input']}` | `{r['shape']}` | "
                 f"{r['abs_err']:.2e} | {r['rel_err']:.2e} | "
                 f"{'PASS' if r['passed'] else 'FAIL'} |")
    L.append("")
    L.append(f"**{n_pass} / {n_total} gradient checks pass.**")
    L.append("")
    L.append("## Negative control")
    L.append("")
    L.append("The same embedding op, but with the buggy backward `dW_e[ids] += dY` instead of "
             "`np.add.at(dW_e, ids, dY)`. The test ids contain repeated tokens, so fancy-indexed "
             "`+=` silently drops gradient - and **only** a gradient check catches it. This op is "
             "*expected to fail*; that it does (missing by `2.49`, not by a noise-floor `1e-9`) "
             "is the proof that the gotcha is real and that the check has teeth.")
    L.append("")
    L.append("| Op | Derivation item | Input checked | Shape | Max abs error | Max rel error | Result |")
    L.append("|----|----|----|----|----|----|----|")
    L.append(f"| {nc_name} | {nc_item} | `W_e` | `{nc_row['shape']}` | {nc_row['abs_err']:.2e} | "
             f"{nc_row['rel_err']:.2e} | {'FAIL (expected)' if nc_ok else 'UNEXPECTED PASS'} |")
    L.append("")
    L.append("## Special checks (consistency, not finite-difference)")
    L.append("")
    L.append("| Check | Derivation item | Metric | Result |")
    L.append("|----|----|----|----|")
    L.append(f"| Fused softmax+CE `==` softmax then CE composed | 10 | "
             f"forward diff {fwd_diff:.2e}, grad diff {grad_diff:.2e} | "
             f"{'PASS' if fused_ok else 'FAIL'} |")
    L.append(f"| Attention key-bias gradient is exactly zero | 15 | "
             f"`|db_K|`={k_mag:.2e} vs `|db_Q|`={q_mag:.2e}, `|db_V|`={v_mag:.2e} | "
             f"{'PASS' if kbias_ok else 'FAIL'} |")
    L.append(f"| AdamW worked example reproduces `theta -> 0.89` | 24 | "
             f"error {worked_err:.2e} | {'PASS' if adamw_ok else 'FAIL'} |")
    L.append(f"| AdamW with `wd=0` equals plain Adam (decoupling) | 24 | "
             f"error {decouple_err:.2e} | {'PASS' if adamw_ok else 'FAIL'} |")
    L.append("")
    L.append("> **The attention key-bias gradient is exactly zero - and that is correct.** "
             "In the fused QKV projection `qkv = x @ W_attn + b_attn`, the bias splits into "
             "`[b_Q | b_K | b_V]`. Adding a constant to *every key* adds the *same* amount to "
             "every score in a given query's row (`S[t,tau] += Q[t].b_K / sqrt(d)`, independent "
             "of the key index `tau`), and softmax is invariant to a constant shift of its "
             "inputs - so the loss does not depend on `b_K` at all, and `db_K` is identically "
             "zero. A pure relative-error check appears to 'fail' on `db_attn` only because it "
             "divides finite-difference noise by finite-difference noise; the absolute error "
             "(`~1e-9`) confirms the formula is right. This is exactly the kind of property a "
             "gradient check is meant to surface.")
    L.append("")
    L.append("## Summary")
    L.append("")

    def _ikey(s):
        n = ''
        for ch in s:
            if ch.isdigit():
                n += ch
            else:
                break
        return (int(n) if n else 999, s)
    items_covered = sorted({r['item'] for r in rows} | {"24"}, key=_ikey)

    L.append(f"- Derivation items exercised: {', '.join(items_covered)} "
             f"(item 1 is the notation convention - nothing to check).")
    L.append("- Every primitive is checked standalone; `sdpa` composes four verified primitives, "
             "`mha` wraps `sdpa` with projections, `transformer block` composes `layernorm` / "
             "`mha` / `linear` / `gelu` with the residual rule, and `tiny GPT` composes the whole "
             "stack including the **two weight-tying accumulation points** on `W_e` (`dW_e` "
             "passes - so both halves of the tied gradient are accumulated correctly).")
    L.append("- Attention is checked at `T=4, H=2, d=8` with `dQ, dK, dV` verified separately.")
    L.append("- Weight matrices use variance-preserving init (`randn / sqrt(fan_in)`) so the "
             "finite-difference check stays well-conditioned through the deep compositions: a "
             "strongly-attenuating init would shrink the deepest gradients below the FD noise "
             "floor and the *check* would fail although the *formula* is correct.")
    L.append("")
    if gate_ok:
        L.append("**Every gradient formula in `derivation.md` is verified correct.** "
                 "Safe to proceed to wiring the ops into an autograd graph.")
    else:
        L.append("**NOT CLEAR** - at least one check did not pass; see the tables above. "
                 "Per the task: stop and find the derivation or code error before proceeding.")
    L.append("")

    with open(RESULTS_MD, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
