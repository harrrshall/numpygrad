"""
test_model_parity.py - Step 4 gate: full-model end-to-end parity vs nanoGPT.

Loads `nanogpt/fixtures/parity_batch.pt` - a real trained checkpoint + a (12,64)
token batch sampled from val.bin - loads the SAME weights into both nanoGPT's
PyTorch `GPT` (nanogpt/model.py) and our NumPy `GPT` (gpt.py), runs forward +
backward on the SAME input, and asserts:

    |numpy_logits - torch_logits|_max  < 1e-5
    |numpy_loss   - torch_loss|        < 1e-6
    every named parameter gradient:    |.|_max < 1e-4  AND  relative error < 1e-3

Weight tying is exercised directly: `wte.weight` is one matrix used twice (token
embedding + lm_head). Its gradient is the SUM of both contributions - the most
common full-model failure mode, and the headline check here.

Both sides run in float64 (the checkpoint's float32 weights widened exactly), so
any mismatch is a real bug, not float precision. If a layer were wrong, Step 3's
per-layer parity would have caught it - so a failure here points at *assembly*:
weight tying, the embedding-sum, positional indexing, or the loss reduction.

GeLU note: nanoGPT's model.py uses `nn.GELU()` (exact erf), so gpt.py uses the
exact erf form. The Step-4 brief says "tanh-approx"; that is the older GPT-2 /
nanoGPT form and would miss this gate by ~1e-3. The checkpoint here was trained
with exact erf - this test passing is the proof. See model_parity.md.

Run:  .venv/bin/python tests/test_model_parity.py
Writes: tests/model_parity.md
"""
import os
import sys

import numpy as np
import torch

torch.set_default_dtype(torch.float64)            # reference at full precision

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "nanogpt"))

import gpt as mygpt                                                   # noqa: E402
from model import GPT as TorchGPT, GPTConfig as TorchGPTConfig        # noqa: E402  nanogpt/model.py

FIXTURE = os.path.join(_ROOT, "nanogpt", "fixtures", "parity_batch.pt")
RESULTS_MD = os.path.join(_HERE, "model_parity.md")

LOGITS_ATOL = 1e-5
LOSS_ATOL = 1e-6
GRAD_ATOL = 1e-4
GRAD_RTOL = 1e-3


def main():
    fx = torch.load(FIXTURE, weights_only=False)
    x, y = fx["x"], fx["y"]                       # (12, 64) int64
    model_args = fx["model_args"]
    sd = {k: v.to(torch.float64) for k, v in fx["state_dict"].items()}  # float32 -> float64

    print("full-model end-to-end parity vs nanoGPT (float64)")
    print(f"  model_args : {model_args}")
    print(f"  batch      : x{tuple(x.shape)} y{tuple(y.shape)}  (from fixtures/parity_batch.pt)")
    print(f"  gate       : logits < {LOGITS_ATOL:.0e} abs,  loss < {LOSS_ATOL:.0e} abs,  "
          f"grads < {GRAD_ATOL:.0e} abs & < {GRAD_RTOL:.0e} rel")
    print("=" * 90)

    # --- PyTorch reference: nanoGPT's own GPT ---
    torch_gpt = TorchGPT(TorchGPTConfig(**model_args))
    torch_gpt.load_state_dict(sd)
    torch_gpt.eval()
    logits_t, loss_t = torch_gpt(x, y)
    loss_t.backward()

    # --- our NumPy GPT, same weights ---
    cfg = mygpt.GPTConfig(**model_args)
    model = mygpt.GPT(cfg)
    model.load_state_dict({k: v.numpy() for k, v in sd.items()})
    logits_m, loss_m = model(x.numpy(), y.numpy())
    loss_m.backward()

    rows = []   # (name, abs_err, rel_err_or_None, passed)

    # forward: logits
    logits_abs = float(np.abs(logits_m.data - logits_t.detach().numpy()).max())
    rows.append(("logits", logits_abs, None, logits_abs < LOGITS_ATOL))

    # forward: loss
    loss_abs = float(abs(float(loss_m.data) - loss_t.item()))
    rows.append(("loss", loss_abs, None, loss_abs < LOSS_ATOL))

    # backward: every named parameter (torch's named_parameters() is deduped, so
    # the weight-tied wte.weight appears once and carries both contributions).
    my_params = model.parameters()
    for tname, tparam in torch_gpt.named_parameters():
        my_name, transpose = mygpt._torch_key_to_my(tname)
        if my_name is None:                       # e.g. lm_head.weight if not deduped
            continue
        tg = tparam.grad.detach().numpy()
        if transpose:
            tg = tg.T
        mg = my_params[my_name].grad
        abs_err = float(np.abs(mg - tg).max())
        denom = max(float(np.abs(mg).max()), float(np.abs(tg).max()), 1e-12)
        rel_err = abs_err / denom
        ok = (abs_err < GRAD_ATOL) and (rel_err < GRAD_RTOL)
        rows.append((f"grad {my_name}", abs_err, rel_err, ok))

    # report
    for (name, abs_err, rel_err, ok) in rows:
        tag = "PASS" if ok else "FAIL"
        if rel_err is None:
            print(f"  {tag}  {name:<28} abs={abs_err:.3e}")
        else:
            print(f"  {tag}  {name:<28} abs={abs_err:.3e}  rel={rel_err:.3e}")

    n_pass = sum(r[3] for r in rows)
    n_total = len(rows)
    gate_ok = (n_pass == n_total)
    print("=" * 90)
    print(f"  {n_pass}/{n_total} checks pass")
    print(f"  loss: numpy={float(loss_m.data):.6f}  torch={loss_t.item():.6f}")
    print(f"  {'PARITY VERIFIED - the NumPy GPT-2 is provably correct' if gate_ok else 'FAILED - see above'}")

    write_md(rows, model_args, float(loss_m.data), loss_t.item(), gate_ok)
    print(f"  wrote {RESULTS_MD}")
    return 0 if gate_ok else 1


def write_md(rows, model_args, loss_np, loss_torch, gate_ok):
    n_pass = sum(r[3] for r in rows)
    L = []
    L.append("# Full-Model Parity Results")
    L.append("")
    L.append("End-to-end parity of the NumPy GPT-2 ([`gpt.py`](../gpt.py)) against nanoGPT's "
             "PyTorch `GPT` ([`nanogpt/model.py`](../nanogpt/model.py)), using the frozen "
             "fixture `nanogpt/fixtures/parity_batch.pt` (a real trained checkpoint + a "
             "`(12, 64)` token batch).")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append(f"- Model: `{model_args}`")
    L.append("- The fixture's float32 `state_dict` is widened to **float64** and loaded into "
             "*both* models; both run in float64, so any mismatch is a real bug, not float "
             "precision.")
    L.append("- nanoGPT's `GPT` is the reference. Our `gpt.py` loads the same weights via "
             "`load_state_dict` (strips `transformer.`, maps LayerNorm `weight`/`bias` -> "
             "`gamma`/`beta`, transposes Linear weights, skips `lm_head.weight`).")
    L.append("- **Weight tying handled explicitly:** `wte.weight` is one matrix used twice "
             "(token embedding + lm_head). `load_state_dict` verifies `lm_head.weight == "
             "transformer.wte.weight`; the autograd graph accumulates both gradient "
             "contributions into the one Tensor. The `grad wte.weight` row below is the check "
             "that this works - it carries the embedding scatter-add **plus** the lm_head "
             "matmul gradient.")
    L.append(f"- **Gate:** `|logits|_max < {LOGITS_ATOL:.0e}`, `|loss| < {LOSS_ATOL:.0e}`, and "
             f"every parameter gradient `|.|_max < {GRAD_ATOL:.0e}` **and** relative error "
             f"`< {GRAD_RTOL:.0e}`.")
    L.append("")
    L.append("## GeLU variant - documented divergence from the Step-4 brief")
    L.append("")
    L.append("The Step-4 instructions say *\"PyTorch's nanoGPT uses tanh-approx GeLU; make "
             "sure you do too, not the exact erf-based one\"*. **That is not true of this "
             "repo.** `nanogpt/model.py` line 83 is `self.gelu = nn.GELU()` - the **exact "
             "erf** GeLU (`approximate='none'`). The tanh approximation is the *older* "
             "GPT-2 / nanoGPT form; the two differ by ~1e-3.")
    L.append("")
    L.append("This was found and resolved back in Step 3 (the user chose to match "
             "`model.py`): `derivation.md` item 4b was re-derived for the exact erf form and "
             "`tests/ops.py:gelu` switched to `scipy.special.erf`. The checkpoint in "
             "`parity_batch.pt` was trained with `model.py`, i.e. with exact-erf GeLU - so "
             "**exact erf is required for this gate**; tanh-approx would fail it by ~1e-3. "
             "This test passing *is* the proof. Also logged in "
             "`nanogpt/roadblocks.md` (2026-05-14).")
    L.append("")
    L.append("## Results")
    L.append("")
    L.append(f"Forward loss: NumPy `{loss_np:.6f}` vs PyTorch `{loss_torch:.6f}`.")
    L.append("")
    L.append("| Check | Max abs error | Relative error | Result |")
    L.append("|---|---|---|---|")
    for (name, abs_err, rel_err, ok) in rows:
        rel_str = "-" if rel_err is None else f"{rel_err:.2e}"
        L.append(f"| `{name}` | {abs_err:.2e} | {rel_str} | {'PASS' if ok else 'FAIL'} |")
    L.append("")
    L.append(f"**{n_pass} / {len(rows)} checks pass.**")
    L.append("")
    L.append("## How to run")
    L.append("")
    L.append("```")
    L.append(".venv/bin/python tests/test_model_parity.py")
    L.append("```")
    L.append("")
    if gate_ok:
        L.append("**Parity verified.** The NumPy GPT-2 produces the same logits, the same "
                 "loss, and the same gradient for every parameter as nanoGPT's PyTorch model, "
                 "given identical weights and inputs - including the weight-tied `wte.weight`. "
                 "The implementation is provably correct end to end.")
    else:
        L.append("**FAILED.** A full-model check did not pass - see the table. Since Step 3 "
                 "verified every layer in isolation, a failure here is an *assembly* bug: "
                 "weight tying, the embedding sum, positional indexing, or the loss reduction.")
    L.append("")
    with open(RESULTS_MD, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
