"""
test_optimizer_parity.py - Step 5 gate: AdamW matches torch.optim.AdamW.

Initialize a parameter tensor identically in NumPy and PyTorch, feed BOTH the
*identical* gradient for 10 steps, and confirm the parameter values match to
1e-6 after every step. A wrong bias-correction term, or weight decay folded into
the gradient instead of decoupled, would diverge immediately.

Hyperparameters are the nanoGPT baseline config
(`nanogpt/config/train_owt_baseline.py`): lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
weight_decay=0.1. Three configs are run:
  - baseline (wd=0.1)        : the gate, exercising the decoupled weight decay.
  - pure Adam (wd=0.0)       : isolates the bias correction (no decay term).
  - alt hyperparams (wd=0.05): different lr/betas, for coverage.

Both sides run in float64.

Run:  .venv/bin/python tests/test_optimizer_parity.py
Writes: tests/optimizer_parity.md
"""
import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from tensor import Tensor          # noqa: E402
from optimizer import AdamW        # noqa: E402

RESULTS_MD = os.path.join(_HERE, "optimizer_parity.md")
STEP_ATOL = 1e-6
N_STEPS = 10


def run_config(lr, betas, eps, weight_decay, seed, shape=(8, 16)):
    """Run our AdamW and torch.optim.AdamW in lock-step for N_STEPS, identical
    init + identical per-step gradients. Returns [(step, max_abs_diff, ok), ...]."""
    rng = np.random.RandomState(seed)
    p0 = rng.randn(*shape)

    my_param = Tensor(p0.copy(), requires_grad=True)
    my_opt = AdamW({'p': my_param}, lr=lr, betas=betas, eps=eps,
                   weight_decay=weight_decay)

    torch_param = torch.tensor(p0.copy(), requires_grad=True)
    torch_opt = torch.optim.AdamW([torch_param], lr=lr, betas=betas, eps=eps,
                                  weight_decay=weight_decay)

    rows = []
    for step in range(1, N_STEPS + 1):
        g = rng.randn(*shape)                       # the SAME gradient for both
        my_param.grad = g.copy()
        torch_param.grad = torch.tensor(g.copy())
        my_opt.step()
        torch_opt.step()
        diff = float(np.abs(my_param.data - torch_param.detach().numpy()).max())
        rows.append((step, diff, diff < STEP_ATOL))
    return rows


CONFIGS = [
    # (name, lr, betas, eps, weight_decay, seed)
    ("baseline (wd=0.1)",       1e-3, (0.9, 0.99),  1e-8, 0.1,  0),
    ("pure Adam (wd=0.0)",      1e-3, (0.9, 0.99),  1e-8, 0.0,  1),
    ("alt hyperparams (wd=0.05)", 3e-4, (0.9, 0.999), 1e-8, 0.05, 2),
]


def main():
    print(f"AdamW parity vs torch.optim.AdamW (float64)  |  "
          f"{N_STEPS} steps, params match < {STEP_ATOL:.0e} after each")
    print("=" * 78)

    all_results = []
    for (name, lr, betas, eps, wd, seed) in CONFIGS:
        rows = run_config(lr, betas, eps, wd, seed)
        all_results.append((name, lr, betas, eps, wd, rows))
        worst = max(d for (_, d, _) in rows)
        ok = all(o for (_, _, o) in rows)
        print(f"\n  {name}   lr={lr}  betas={betas}  eps={eps}  wd={wd}   "
              f"{'PASS' if ok else 'FAIL'}")
        for (step, diff, step_ok) in rows:
            print(f"    {'PASS' if step_ok else 'FAIL'}  step {step:2d}   "
                  f"max|diff| = {diff:.2e}")
        print(f"    worst over {N_STEPS} steps: {worst:.2e}")

    n_pass = sum(o for (_, _, _, _, _, rows) in all_results for (_, _, o) in rows)
    n_total = sum(len(rows) for (_, _, _, _, _, rows) in all_results)
    gate_ok = (n_pass == n_total)
    print("\n" + "=" * 78)
    print(f"  {n_pass}/{n_total} step checks pass (< {STEP_ATOL:.0e})")
    print(f"  {'AdamW MATCHES torch.optim.AdamW' if gate_ok else 'FAILED - investigate above'}")

    write_md(all_results, gate_ok)
    print(f"  wrote {RESULTS_MD}")
    return 0 if gate_ok else 1


def write_md(all_results, gate_ok):
    n_pass = sum(o for (_, _, _, _, _, rows) in all_results for (_, _, o) in rows)
    n_total = sum(len(rows) for (_, _, _, _, _, rows) in all_results)

    L = []
    L.append("# Optimizer Parity Results")
    L.append("")
    L.append("Parity of [`optimizer.py`](../optimizer.py)'s `AdamW` against PyTorch's "
             "`torch.optim.AdamW`.")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append(f"- A parameter tensor is initialized **identically** in NumPy and PyTorch.")
    L.append(f"- For **{N_STEPS} steps**, BOTH optimizers are given the **same** gradient, "
             f"then `step()` is called on each.")
    L.append(f"- After every step the parameter values must match: "
             f"`max|p_numpy - p_torch| < {STEP_ATOL:.0e}`.")
    L.append("- Both run in float64. Hyperparameters are the nanoGPT baseline "
             "(`config/train_owt_baseline.py`): `lr=1e-3, betas=(0.9, 0.99), eps=1e-8, "
             "weight_decay=0.1`.")
    L.append("")
    L.append("## AdamW: decoupled weight decay")
    L.append("")
    L.append("AdamW differs from Adam-with-L2 in *where* the weight decay is applied. "
             "`optimizer.py` does it the decoupled way - directly on the parameter, "
             "**not** folded into the gradient:")
    L.append("")
    L.append("```")
    L.append("p      *= (1 - lr * weight_decay)            # decoupled decay (the AdamW part)")
    L.append("m       = beta1*m + (1-beta1)*g              # 1st-moment EMA")
    L.append("v       = beta2*v + (1-beta2)*g**2           # 2nd-moment EMA")
    L.append("p      -= lr * (m / (1-beta1**t)) / (sqrt(v / (1-beta2**t)) + eps)")
    L.append("```")
    L.append("")
    L.append("`m`/`v` start at zero, so the `1 - beta**t` bias correction undoes the "
             "cold-start bias. The `pure Adam (wd=0.0)` config below isolates that bias "
             "correction; the `wd=0.1` configs additionally exercise the decoupled decay. "
             "Folding `weight_decay * p` into the gradient (the common bug) would fail the "
             "`wd>0` configs immediately.")
    L.append("")
    L.append("## Results")
    L.append("")
    for (name, lr, betas, eps, wd, rows) in all_results:
        worst = max(d for (_, d, _) in rows)
        cfg_ok = all(o for (_, _, o) in rows)
        L.append(f"### {name}")
        L.append("")
        L.append(f"`lr={lr}`, `betas={betas}`, `eps={eps}`, `weight_decay={wd}` - "
                 f"**{'PASS' if cfg_ok else 'FAIL'}** (worst over {N_STEPS} steps: "
                 f"`{worst:.2e}`)")
        L.append("")
        L.append("| Step | max&#124;p_numpy - p_torch&#124; | Result |")
        L.append("|---|---|---|")
        for (step, diff, ok) in rows:
            L.append(f"| {step} | {diff:.2e} | {'PASS' if ok else 'FAIL'} |")
        L.append("")
    L.append(f"**{n_pass} / {n_total} step checks pass** (`< {STEP_ATOL:.0e}`).")
    L.append("")
    L.append("## How to run")
    L.append("")
    L.append("```")
    L.append(".venv/bin/python tests/test_optimizer_parity.py")
    L.append("```")
    L.append("")
    if gate_ok:
        L.append("**`AdamW` matches `torch.optim.AdamW` numerically.** Bias correction and "
                 "decoupled weight decay are both correct - ready for the training loop.")
    else:
        L.append("**FAILED** - a step diverged; see the table. A wrong bias correction or a "
                 "weight-decay term folded into the gradient is the usual cause.")
    L.append("")
    with open(RESULTS_MD, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
