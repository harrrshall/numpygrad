"""
test_train_step_parity.py - Step 6 gate: one full training step matches PyTorch.

The final correctness check before training. Starting from the SAME checkpoint
(`nanogpt/fixtures/parity_batch.pt`), run ONE full training step on both sides
and compare the resulting weights:

    forward(x, y) -> loss   ->   loss.backward()   ->   optimizer.step()

PyTorch side : nanoGPT's `GPT` + `configure_optimizers` (its real 2-group AdamW:
               weight decay on >=2D tensors, 0.0 on 1D tensors).
NumPy side   : `gpt.py` `GPT` + `optimizer.py` `AdamW`, same grouping via
               `no_decay = {1D parameter names}`, same hyperparameters.

Gate: every parameter matches to 1e-5 after the step.

Both sides use a FRESH optimizer (m=v=0, step 1) - the fixture carries no
optimizer state - and the nanoGPT baseline hyperparameters (lr=1e-3,
betas=(0.9,0.99), eps=1e-8, weight_decay=0.1). Gradient clipping
(nanoGPT's grad_clip=1.0) is a training-LOOP operation applied *before*
`optimizer.step()`, not part of the optimizer; it is left for the training loop
(Step 7). The global grad norm is reported below so it's transparent whether a
clipped step would have differed; the test skips clipping on BOTH sides, so the
comparison is exact regardless.

This step composes Step 4 (full-model fwd/bwd parity, verified) and Step 5
(AdamW parity, verified). If it fails, the culprit is the *composition* -
optimizer grouping, or a stale gradient/state - since each half is already
verified in isolation.

Run:  .venv/bin/python tests/test_train_step_parity.py
Writes: tests/train_step_parity.md
"""
import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "nanogpt"))

import gpt as mygpt                                                   # noqa: E402
from optimizer import AdamW                                          # noqa: E402
from model import GPT as TorchGPT, GPTConfig as TorchGPTConfig       # noqa: E402  nanogpt/model.py

FIXTURE = os.path.join(_ROOT, "nanogpt", "fixtures", "parity_batch.pt")
RESULTS_MD = os.path.join(_HERE, "train_step_parity.md")

LR = 1e-3
BETAS = (0.9, 0.99)
EPS = 1e-8
WEIGHT_DECAY = 0.1
PARAM_ATOL = 1e-5


def main():
    fx = torch.load(FIXTURE, weights_only=False)
    x, y = fx["x"], fx["y"]
    model_args = fx["model_args"]
    sd = {k: v.to(torch.float64) for k, v in fx["state_dict"].items()}

    print("single-step training parity vs nanoGPT (float64)")
    print(f"  model_args : {model_args}")
    print(f"  optimizer  : AdamW lr={LR} betas={BETAS} eps={EPS} weight_decay={WEIGHT_DECAY} "
          f"(decay on >=2D tensors only); fresh state, step 1")
    print(f"  gate       : every parameter matches < {PARAM_ATOL:.0e} after one step")
    print("=" * 92)

    # ---- PyTorch: one full training step from the checkpoint ----
    torch_gpt = TorchGPT(TorchGPTConfig(**model_args))
    torch_gpt.load_state_dict(sd)
    torch_before = {n: p.detach().numpy().copy() for n, p in torch_gpt.named_parameters()}
    torch_opt = torch_gpt.configure_optimizers(WEIGHT_DECAY, LR, BETAS, "cpu")  # nanoGPT's own 2-group AdamW
    _, loss_t = torch_gpt(x, y)
    loss_t.backward()
    grad_norm = float(torch.cat([p.grad.flatten() for p in torch_gpt.parameters()]).norm())
    torch_opt.step()
    torch_after = {n: p.detach().numpy().copy() for n, p in torch_gpt.named_parameters()}

    # ---- NumPy: one full training step from the SAME checkpoint ----
    model = mygpt.GPT(mygpt.GPTConfig(**model_args))
    model.load_state_dict({k: v.numpy() for k, v in sd.items()})
    params = model.parameters()
    no_decay = {n for n, p in params.items() if p.data.ndim < 2}        # nanoGPT: 1D tensors excluded
    my_opt = AdamW(params, lr=LR, betas=BETAS, eps=EPS,
                   weight_decay=WEIGHT_DECAY, no_decay=no_decay)
    _, loss_m = model(x.numpy(), y.numpy())
    loss_m.backward()
    my_opt.step()

    # ---- compare resulting weights ----
    rows = []
    for tname, tparam in torch_gpt.named_parameters():
        my_name, transpose = mygpt._torch_key_to_my(tname)
        if my_name is None:
            continue
        ta_after = torch_after[tname]
        ta_before = torch_before[tname]
        if transpose:
            ta_after, ta_before = ta_after.T, ta_before.T
        ma_after = params[my_name].data
        diff = float(np.abs(ma_after - ta_after).max())
        update = float(np.abs(ta_after - ta_before).max())             # how much the step moved it
        decayed = my_name not in no_decay
        rows.append((my_name, update, diff, diff < PARAM_ATOL, decayed))

    for (name, update, diff, ok, decayed) in rows:
        grp = "decay  " if decayed else "no-decay"
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<22} [{grp}]  "
              f"step moved |.|<= {update:.2e}   parity |.|<= {diff:.2e}")

    n_pass = sum(r[3] for r in rows)
    n_total = len(rows)
    gate_ok = (n_pass == n_total)
    print("=" * 92)
    print(f"  loss: numpy={float(loss_m.data):.6f}  torch={loss_t.item():.6f}   "
          f"global grad norm={grad_norm:.4f}  (grad_clip=1.0 would "
          f"{'clip' if grad_norm > 1.0 else 'be a no-op'})")
    print(f"  {n_pass}/{n_total} parameters match < {PARAM_ATOL:.0e} after one full training step")
    print(f"  {'TRAINING-STEP PARITY VERIFIED - ready to train' if gate_ok else 'FAILED - see above'}")

    write_md(rows, model_args, float(loss_m.data), loss_t.item(), grad_norm, gate_ok)
    print(f"  wrote {RESULTS_MD}")
    return 0 if gate_ok else 1


def write_md(rows, model_args, loss_np, loss_torch, grad_norm, gate_ok):
    n_pass = sum(r[3] for r in rows)
    L = []
    L.append("# Single-Step Training Parity Results")
    L.append("")
    L.append("The final correctness check before training. From the same checkpoint "
             "(`nanogpt/fixtures/parity_batch.pt`), one **full training step** -")
    L.append("")
    L.append("```")
    L.append("forward(x, y) -> loss   ->   loss.backward()   ->   optimizer.step()")
    L.append("```")
    L.append("")
    L.append("is run on both sides and the resulting weights are compared.")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append(f"- Model: `{model_args}`")
    L.append("- **PyTorch side:** nanoGPT's `GPT` + `configure_optimizers` - its own "
             "2-group `torch.optim.AdamW` (weight decay on >=2D tensors, `0.0` on 1D "
             "tensors).")
    L.append("- **NumPy side:** `gpt.py` `GPT` + `optimizer.py` `AdamW`, the same grouping "
             "via `no_decay = {1D parameter names}`, the same hyperparameters.")
    L.append(f"- Hyperparameters (nanoGPT baseline): `lr={LR}`, `betas={BETAS}`, `eps={EPS}`, "
             f"`weight_decay={WEIGHT_DECAY}`. Both optimizers start **fresh** (m=v=0, step 1) "
             "- the fixture carries no optimizer state.")
    L.append("- Both run in float64. Linear weights are transposed between the layouts when "
             "comparing.")
    L.append(f"- **Gate:** every parameter matches to `{PARAM_ATOL:.0e}` after the step.")
    L.append("")
    L.append("## Gradient clipping")
    L.append("")
    L.append(f"The global gradient norm for this batch is **{grad_norm:.4f}**. nanoGPT's "
             "training loop applies `clip_grad_norm_(.., grad_clip=1.0)` *before* "
             "`optimizer.step()` - a training-LOOP operation, not part of the optimizer - so "
             "it is left for Step 7. This test skips clipping on **both** sides, so the "
             "comparison is exact regardless; the norm is reported here only for "
             f"transparency (`grad_clip=1.0` would "
             f"{'**clip** in a real nanoGPT step' if grad_norm > 1.0 else 'be a no-op here'}).")
    L.append("")
    L.append("## Results")
    L.append("")
    L.append(f"Forward loss (sanity, matches Step 4): NumPy `{loss_np:.6f}` vs PyTorch "
             f"`{loss_torch:.6f}`.")
    L.append("")
    L.append("`step moved` = how far the optimizer moved that parameter (max abs), so the "
             "`1e-5` gate isn't passed trivially by nothing changing. `parity` = "
             "`max|p_numpy - p_torch|` after the step.")
    L.append("")
    L.append("| Parameter | Group | step moved | parity (max abs diff) | Result |")
    L.append("|---|---|---|---|---|")
    for (name, update, diff, ok, decayed) in rows:
        grp = "decay" if decayed else "no-decay"
        L.append(f"| `{name}` | {grp} | {update:.2e} | {diff:.2e} | "
                 f"{'PASS' if ok else 'FAIL'} |")
    L.append("")
    L.append(f"**{n_pass} / {len(rows)} parameters match** to `{PARAM_ATOL:.0e}` after one "
             "full training step.")
    L.append("")
    L.append("## How to run")
    L.append("")
    L.append("```")
    L.append(".venv/bin/python tests/test_train_step_parity.py")
    L.append("```")
    L.append("")
    if gate_ok:
        L.append("**Training-step parity verified.** One full step - forward, backward, "
                 "AdamW update - of the NumPy GPT-2 lands on the same weights as nanoGPT's "
                 "PyTorch step, for every parameter, including the weight-tied `wte.weight` "
                 "and the decay / no-decay grouping. The implementation is correct end to "
                 "end and ready to train.")
    else:
        L.append("**FAILED** - a parameter diverged after one step. Step 4 (model fwd/bwd) "
                 "and Step 5 (AdamW) are each verified in isolation, so a failure here is in "
                 "the *composition*: the optimizer's decay/no-decay grouping, or a stale "
                 "gradient / optimizer state.")
    L.append("")
    with open(RESULTS_MD, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
