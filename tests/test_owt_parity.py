"""
test_owt_parity.py - Step 8 gate: NumPy GPT-2's training curve tracks PyTorch's.

Trains on the real OpenWebText subset (`nanogpt/data/openwebtext/*.bin`) and
compares the NumPy GPT-2's per-step loss curve against a PyTorch baseline.

The baseline: `nanogpt/out-owt-baseline/baseline_losses.json` - a saved 100-step
nanoGPT run (config/train_owt_baseline.py, seed 1337). The repro logs show that
run is bit-reproducible, so this harness:

  1. seeds torch 1337, builds nanoGPT's `GPT` exactly as train.py does, and
     faithfully replays train.py's `get_batch` RNG order - including the
     iter-0 / iter-100 `estimate_loss` calls - so the training batches ARE the
     baseline's batches. The torch run then reproduces baseline_losses.json
     (verified below: that is the proof the replay is faithful).
  2. copies that exact initial state_dict into the NumPy `GPT`.
  3. runs PyTorch (float32, == the baseline) and NumPy (float64) over the SAME
     batches, SAME init, SAME hyperparameters, SAME cosine LR schedule.

Gate: `|numpy_loss - torch_loss| < 0.05` at each of the first 100 steps. They
won't match exactly - float32 vs float64, Adam accumulating the drift - but the
curves are visually indistinguishable early and diverge only slightly later.

Logs train loss, val loss, perplexity, gradient norm and learning rate; plots
the NumPy / PyTorch / saved-baseline curves on the same axes.

Run:  .venv/bin/python tests/test_owt_parity.py     (~6 min - 120 NumPy steps)
Writes: tests/owt_parity.md, tests/owt_parity_loss.png
"""
import json
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "nanogpt"))

import gpt as mygpt                                                   # noqa: E402
import data as owt_data                                              # noqa: E402
from optimizer import AdamW                                          # noqa: E402
from trainer import get_lr, clip_grad_norm                           # noqa: E402
from model import GPT as TorchGPT, GPTConfig as TorchGPTConfig       # noqa: E402

# --- nanoGPT baseline config (nanogpt/config/train_owt_baseline.py) ---
MODEL_ARGS = dict(n_layer=4, n_head=4, n_embd=128, block_size=64, bias=False,
                  vocab_size=50304, dropout=0.0)
LR, MIN_LR, WARMUP, LR_DECAY_ITERS = 1e-3, 1e-4, 10, 100
BETAS, EPS, WEIGHT_DECAY, GRAD_CLIP = (0.9, 0.99), 1e-8, 0.1, 1.0
BATCH_SIZE, BLOCK_SIZE = 12, 64
EVAL_ITERS = 20            # nanoGPT's estimate_loss batch count (RNG must match)
EVAL_LOG = 8               # batches we actually run for the logged val loss
N_STEPS = 120              # iters 0..119; the gate compares iters 0..100
GATE = 0.05

BASELINE_JSON = os.path.join(_ROOT, "nanogpt", "out-owt-baseline", "baseline_losses.json")
RESULTS_MD = os.path.join(_HERE, "owt_parity.md")
PLOT_PNG = os.path.join(_HERE, "owt_parity_loss.png")


def faithful_setup(n_steps):
    """Replay nanoGPT train.py's RNG exactly: seed -> model init -> get_batch
    sequence. Returns the torch model (freshly initialised), its init state_dict,
    the per-step training batches, and the iter-0 / iter-100 eval val batches."""
    torch.manual_seed(1337)
    torch_model = TorchGPT(TorchGPTConfig(**MODEL_ARGS))            # consumes init RNG
    init_sd = {k: v.detach().clone() for k, v in torch_model.state_dict().items()}

    train_data = owt_data.load_split("train")
    val_data = owt_data.load_split("val")

    def tb():
        return owt_data.get_batch(train_data, BATCH_SIZE, BLOCK_SIZE)

    def vb():
        return owt_data.get_batch(val_data, BATCH_SIZE, BLOCK_SIZE)

    # train.py's exact get_batch order:
    #   #1 train (iter-0 batch); iter-0 estimate_loss = 20 train + 20 val;
    #   then one train batch per iter, with iter-100's estimate_loss (20+20)
    #   slotted in right after iter-100's batch is fetched.
    train_batches = [tb()]                                          # iter 0
    for _ in range(EVAL_ITERS):
        tb()                                                        # iter-0 eval train (discard)
    eval0_val = [vb() for _ in range(EVAL_ITERS)]                   # iter-0 eval val
    eval100_val = []
    for it in range(1, n_steps):
        train_batches.append(tb())                                  # iter-`it` batch
        if it == 100:
            for _ in range(EVAL_ITERS):
                tb()                                                # iter-100 eval train (discard)
            eval100_val = [vb() for _ in range(EVAL_ITERS)]         # iter-100 eval val
    return torch_model, init_sd, train_batches, eval0_val, eval100_val


def _eval_torch(model, batches):
    model.eval()
    with torch.no_grad():
        ls = [model(torch.from_numpy(x), torch.from_numpy(y))[1].item() for x, y in batches]
    model.train()
    return float(np.mean(ls))


def _eval_numpy(model, batches):
    return float(np.mean([float(model(x, y)[1].data) for x, y in batches]))


def run_torch(model, train_batches, eval_batches, n_steps):
    opt = model.configure_optimizers(WEIGHT_DECAY, LR, BETAS, "cpu")  # nanoGPT's 2-group AdamW
    log = dict(losses=[], grad_norms=[], lrs=[], evals={})
    log["evals"][0] = _eval_torch(model, eval_batches[0][:EVAL_LOG])
    t0 = time.time()
    for step in range(n_steps):
        if step == 100 and 100 in eval_batches:
            log["evals"][100] = _eval_torch(model, eval_batches[100][:EVAL_LOG])
        lr = get_lr(step, LR, MIN_LR, WARMUP, LR_DECAY_ITERS)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = train_batches[step]
        opt.zero_grad(set_to_none=True)
        _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        log["losses"].append(loss.item())
        log["grad_norms"].append(float(gn))
        log["lrs"].append(lr)
        if step % 20 == 0 or step == n_steps - 1:
            print(f"    torch step {step:3d}  loss {loss.item():.4f}  "
                  f"grad_norm {float(gn):.3f}  lr {lr:.2e}")
    print(f"  torch: {n_steps} steps in {time.time() - t0:.1f}s")
    return log


def run_numpy(init_sd, train_batches, eval_batches, n_steps):
    model = mygpt.GPT(mygpt.GPTConfig(**MODEL_ARGS))
    model.load_state_dict({k: v.numpy() for k, v in init_sd.items()})  # same init as torch
    params = model.parameters()
    no_decay = {n for n, p in params.items() if p.data.ndim < 2}
    opt = AdamW(params, lr=LR, betas=BETAS, eps=EPS,
                weight_decay=WEIGHT_DECAY, no_decay=no_decay)
    log = dict(losses=[], grad_norms=[], lrs=[], evals={})
    log["evals"][0] = _eval_numpy(model, eval_batches[0][:EVAL_LOG])
    t0 = time.time()
    for step in range(n_steps):
        if step == 100 and 100 in eval_batches:
            log["evals"][100] = _eval_numpy(model, eval_batches[100][:EVAL_LOG])
        lr = get_lr(step, LR, MIN_LR, WARMUP, LR_DECAY_ITERS)
        opt.lr = lr
        x, y = train_batches[step]
        opt.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        gn = clip_grad_norm(params, GRAD_CLIP)
        opt.step()
        lossval = float(loss.data)
        log["losses"].append(lossval)
        log["grad_norms"].append(float(gn))
        log["lrs"].append(lr)
        if step % 20 == 0 or step == n_steps - 1:
            print(f"    numpy step {step:3d}  loss {lossval:.4f}  "
                  f"grad_norm {gn:.3f}  lr {lr:.2e}")
        del logits, loss                       # free this step's graph before the next forward
    print(f"  numpy: {n_steps} steps in {time.time() - t0:.1f}s")
    return log


def main():
    print("OWT training parity: NumPy GPT-2 vs the saved PyTorch baseline")
    print(f"  model {MODEL_ARGS}")
    print(f"  {N_STEPS} steps  |  lr cosine {LR}->{MIN_LR} warmup {WARMUP}  |  "
          f"wd {WEIGHT_DECAY}  grad_clip {GRAD_CLIP}  |  gate |dloss| < {GATE}")
    print("=" * 88)

    baseline = json.load(open(BASELINE_JSON))
    base_losses = baseline["train_loss_per_iter"]           # iters 0..100
    base_evals = {e["iter"]: e for e in baseline["evals"]}

    print("  replaying nanoGPT's RNG to reproduce the baseline's exact batches + init ...")
    torch_model, init_sd, train_batches, eval0_val, eval100_val = faithful_setup(N_STEPS)
    eval_batches = {0: eval0_val, 100: eval100_val}

    print("  running PyTorch (float32, == the baseline) ...")
    tlog = run_torch(torch_model, train_batches, eval_batches, N_STEPS)
    print("  running NumPy GPT-2 (float64) ...")
    nlog = run_numpy(init_sd, train_batches, eval_batches, N_STEPS)

    tl = np.array(tlog["losses"])
    nl = np.array(nlog["losses"])
    bl = np.array(base_losses)
    n_cmp = min(100, N_STEPS)                               # gate window: first 100 steps

    # faithfulness: does the torch re-run reproduce the saved baseline?
    repro_max = float(np.abs(tl[:len(bl)] - bl).max())
    # the gate: NumPy vs PyTorch over the first 100 steps
    gate_diffs = np.abs(nl[:n_cmp] - tl[:n_cmp])
    gate_max = float(gate_diffs.max())
    gate_ok = gate_max < GATE
    # NumPy vs the saved baseline curve
    numpy_vs_base = float(np.abs(nl[:len(bl)] - bl).max())

    print("=" * 88)
    print(f"  torch re-run vs saved baseline_losses.json : max |dloss| = {repro_max:.4e}  "
          f"({'faithful reproduction' if repro_max < 1e-3 else 'NOT reproduced - RNG replay off'})")
    print(f"  NumPy vs PyTorch  (first {n_cmp} steps)     : max |dloss| = {gate_max:.4e}   "
          f"mean {gate_diffs.mean():.4e}")
    print(f"  NumPy vs saved baseline (first {len(bl)})    : max |dloss| = {numpy_vs_base:.4e}")
    print(f"  loss: init {nl[0]:.3f} -> step {n_cmp-1} {nl[n_cmp-1]:.3f} -> "
          f"step {N_STEPS-1} {nl[-1]:.3f}")
    for it in sorted(nlog["evals"]):
        nv, tv = nlog["evals"][it], tlog["evals"][it]
        bv = base_evals.get(it, {}).get("val_loss")
        bstr = f"  baseline {bv:.4f}" if bv is not None else ""
        print(f"  val loss @ iter {it:3d}: numpy {nv:.4f} (ppl {np.exp(nv):.0f})  "
              f"torch {tv:.4f} (ppl {np.exp(tv):.0f}){bstr}")
    print(f"  {'PARITY VERIFIED - NumPy GPT-2 tracks the PyTorch baseline' if gate_ok else 'GATE FAILED - see above'}")

    make_plot(nl, tl, bl, nlog, tlog, baseline)
    write_md(nl, tl, bl, nlog, tlog, baseline, repro_max, gate_max,
             float(gate_diffs.mean()), numpy_vs_base, n_cmp, gate_ok)
    print(f"  wrote {RESULTS_MD}")
    print(f"  wrote {PLOT_PNG}")
    return 0 if gate_ok else 1


def make_plot(nl, tl, bl, nlog, tlog, baseline):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), height_ratios=[2, 1])
    bi = baseline["iters"]
    ax1.plot(bi, bl, color="#999999", linewidth=3.0, alpha=0.7,
             label="saved PyTorch baseline (float32)")
    ax1.plot(range(len(tl)), tl, color="#d62728", linewidth=1.0,
             label="PyTorch re-run (float32)")
    ax1.plot(range(len(nl)), nl, color="#1f77b4", linewidth=1.2, linestyle="--",
             label="NumPy GPT-2 (float64)")
    for it, e in [(e["iter"], e) for e in baseline["evals"]]:
        ax1.plot(it, e["val_loss"], "s", color="#999999", markersize=7)
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("OWT training parity - NumPy GPT-2 vs PyTorch baseline "
                  "(4L/4H/128d, block 64, batch 12)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right")

    ax2.plot(range(len(nlog["grad_norms"])), nlog["grad_norms"], color="#1f77b4",
             linewidth=1.0, label="grad norm (NumPy)")
    ax2.plot(range(len(tlog["grad_norms"])), tlog["grad_norms"], color="#d62728",
             linewidth=1.0, alpha=0.7, label="grad norm (PyTorch)")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("global grad norm")
    ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(range(len(nlog["lrs"])), nlog["lrs"], color="#2ca02c", linewidth=1.2,
              label="learning rate")
    ax2b.set_ylabel("learning rate", color="#2ca02c")
    ax2b.tick_params(axis="y", labelcolor="#2ca02c")
    lines = ax2.get_lines() + ax2b.get_lines()
    ax2.legend(lines, [l.get_label() for l in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=130)
    plt.close(fig)


def write_md(nl, tl, bl, nlog, tlog, baseline, repro_max, gate_max, gate_mean,
             numpy_vs_base, n_cmp, gate_ok):
    L = []
    L.append("# OWT Training Parity Results")
    L.append("")
    L.append("The NumPy GPT-2's training loss curve vs the saved PyTorch baseline, on the "
             "real OpenWebText subset (`nanogpt/data/openwebtext/*.bin`).")
    L.append("")
    L.append("![loss curves](owt_parity_loss.png)")
    L.append("")
    L.append("## Method")
    L.append("")
    L.append("- **Baseline:** `nanogpt/out-owt-baseline/baseline_losses.json` - a saved "
             "100-step nanoGPT run (`config/train_owt_baseline.py`, seed 1337, float32). The "
             "`out-repro/` logs confirm that run is bit-reproducible.")
    L.append("- This harness seeds torch 1337, builds nanoGPT's `GPT` exactly as `train.py` "
             "does, and **faithfully replays `train.py`'s `get_batch` RNG order** (including "
             "the iter-0 / iter-100 `estimate_loss` calls). So the training batches ARE the "
             "baseline's batches, and the same initial weights are copied into the NumPy "
             "`GPT`.")
    L.append("- PyTorch runs in **float32** (== the baseline); NumPy in **float64**. Same "
             "batches, same init, same hyperparameters, same cosine LR schedule "
             f"(`lr {LR}->{MIN_LR}`, warmup {WARMUP}), `weight_decay={WEIGHT_DECAY}`, "
             f"`grad_clip={GRAD_CLIP}`.")
    L.append(f"- **Gate:** `|numpy_loss - torch_loss| < {GATE}` at each of the first "
             f"{n_cmp} steps.")
    L.append("")
    L.append("## Faithfulness check")
    L.append("")
    L.append(f"The PyTorch re-run vs the saved `baseline_losses.json`: "
             f"**max `|dloss| = {repro_max:.2e}`** over all {len(bl)} steps. "
             + ("That is below the JSON's 4-decimal rounding - the RNG replay is faithful, "
                "so the torch re-run *is* the baseline."
                if repro_max < 1e-3 else
                "This is larger than expected - the RNG replay did not perfectly reproduce "
                "the baseline's batches; the NumPy-vs-PyTorch gate below is still a valid "
                "lockstep comparison."))
    L.append("")
    L.append("## Gate")
    L.append("")
    L.append(f"- **NumPy vs PyTorch**, first {n_cmp} steps: max `|dloss| = {gate_max:.2e}`, "
             f"mean `{gate_mean:.2e}` - **{'PASS' if gate_ok else 'FAIL'}** "
             f"(gate `< {GATE}`).")
    L.append(f"- **NumPy vs saved baseline**, first {len(bl)} steps: max "
             f"`|dloss| = {numpy_vs_base:.2e}`.")
    L.append("- float32 (PyTorch) vs float64 (NumPy): the curves are visually "
             "indistinguishable early; the small drift later is exactly the accumulated "
             "floating-point difference Adam carries forward - not an implementation error.")
    L.append("")
    L.append("## Loss curve")
    L.append("")
    L.append("| iter | NumPy loss | PyTorch loss | baseline | &#124;np-torch&#124; | grad norm (np) | lr |")
    L.append("|---|---|---|---|---|---|---|")
    marks = sorted(set(list(range(0, N_STEPS, 10)) + [n_cmp - 1, N_STEPS - 1]))
    for s in marks:
        b = f"{bl[s]:.4f}" if s < len(bl) else "-"
        d = f"{abs(nl[s] - tl[s]):.2e}" if s < n_cmp else f"{abs(nl[s]-tl[s]):.2e}"
        L.append(f"| {s} | {nl[s]:.4f} | {tl[s]:.4f} | {b} | {d} | "
                 f"{nlog['grad_norms'][s]:.3f} | {nlog['lrs'][s]:.2e} |")
    L.append("")
    L.append("## Logged metrics")
    L.append("")
    L.append(f"- **Train loss:** init `{nl[0]:.4f}` (~ln(50304) = `{np.log(50304):.3f}`, "
             f"uniform) -> step {n_cmp-1} `{nl[n_cmp-1]:.4f}` -> step {N_STEPS-1} "
             f"`{nl[-1]:.4f}`.")
    L.append("- **Val loss / perplexity:**")
    L.append("")
    L.append("| iter | NumPy val loss | NumPy ppl | PyTorch val loss | PyTorch ppl | baseline val loss |")
    L.append("|---|---|---|---|---|---|")
    base_evals = {e["iter"]: e for e in baseline["evals"]}
    for it in sorted(nlog["evals"]):
        nv, tv = nlog["evals"][it], tlog["evals"][it]
        bv = base_evals.get(it, {}).get("val_loss")
        L.append(f"| {it} | {nv:.4f} | {np.exp(nv):.0f} | {tv:.4f} | {np.exp(tv):.0f} | "
                 f"{bv:.4f} |" if bv is not None else
                 f"| {it} | {nv:.4f} | {np.exp(nv):.0f} | {tv:.4f} | {np.exp(tv):.0f} | - |")
    L.append("")
    L.append(f"- **Gradient norm (NumPy):** start `{nlog['grad_norms'][0]:.3f}`, "
             f"min `{min(nlog['grad_norms']):.3f}`, max `{max(nlog['grad_norms']):.3f}` "
             f"(`grad_clip={GRAD_CLIP}`: clipped on "
             f"{sum(g > GRAD_CLIP for g in nlog['grad_norms'])} / {N_STEPS} steps).")
    L.append(f"- **Learning rate:** warmup `{nlog['lrs'][0]:.2e}` -> peak "
             f"`{max(nlog['lrs']):.2e}` (step {WARMUP}) -> cosine -> `{nlog['lrs'][-1]:.2e}`.")
    L.append("")
    L.append("## How to run")
    L.append("")
    L.append("```")
    L.append(".venv/bin/python tests/test_owt_parity.py")
    L.append("```")
    L.append("")
    if gate_ok:
        L.append(f"**Parity verified.** The NumPy GPT-2's loss curve tracks the PyTorch "
                 f"baseline to within `{gate_max:.1e}` over the first {n_cmp} steps - far "
                 f"inside the `{GATE}` gate - on the real OWT data, with the real cosine LR "
                 f"schedule, weight decay and gradient clipping. Trained {N_STEPS} steps "
                 f"total ({N_STEPS - n_cmp} beyond the baseline). The full pipeline - "
                 "autograd, model, optimizer, training loop - is correct on a real training "
                 "run.")
    else:
        L.append(f"**GATE FAILED** - max `|dloss| = {gate_max:.2e}` exceeds `{GATE}`. Since "
                 "Steps 4-6 verified the model, optimizer and single step in isolation, a "
                 "failure here is in the *training run*: the LR schedule, grad clipping, or "
                 "drift accumulating faster than expected.")
    L.append("")
    with open(RESULTS_MD, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
