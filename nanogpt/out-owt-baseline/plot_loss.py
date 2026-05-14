"""
Parse train.log into a loss curve (baseline_losses.json + baseline_loss_curve.png).
"""
import json
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
LOG = HERE / "train.log"

iter_re = re.compile(r"^iter (\d+): loss ([\d.]+), time ([\d.]+)ms")
eval_re = re.compile(
    r"^step (\d+): train loss ([\d.]+), val loss ([\d.]+)"
)

iters = []
losses = []
times_ms = []
evals = []

with LOG.open() as f:
    for line in f:
        m = iter_re.match(line)
        if m:
            iters.append(int(m.group(1)))
            losses.append(float(m.group(2)))
            times_ms.append(float(m.group(3)))
            continue
        m = eval_re.match(line)
        if m:
            evals.append({
                "iter": int(m.group(1)),
                "train_loss": float(m.group(2)),
                "val_loss": float(m.group(3)),
            })

artifact = {
    "iters": iters,
    "train_loss_per_iter": losses,
    "iter_time_ms": times_ms,
    "evals": evals,
    "summary": {
        "n_iters": len(iters),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "median_iter_ms": (
            sorted(times_ms)[len(times_ms) // 2] if times_ms else None
        ),
    },
}

with (HERE / "baseline_losses.json").open("w") as f:
    json.dump(artifact, f, indent=2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(iters, losses, label="train loss (per step)", color="#1f77b4",
        linewidth=1.2)

# Overlay eval points
if evals:
    ev_iters = [e["iter"] for e in evals]
    ev_train = [e["train_loss"] for e in evals]
    ev_val = [e["val_loss"] for e in evals]
    ax.plot(ev_iters, ev_train, "o", color="#1f77b4",
            label="train loss (eval, 20-batch mean)", markersize=8)
    ax.plot(ev_iters, ev_val, "s", color="#d62728",
            label="val loss (eval, 20-batch mean)", markersize=8)

ax.set_xlabel("iteration")
ax.set_ylabel("cross-entropy loss")
ax.set_title(
    "nanoGPT baseline — 4L/4H/128d, block=64, batch=12, OWT subset (5.6M tok), CPU"
)
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(HERE / "baseline_loss_curve.png", dpi=140)
print("wrote", HERE / "baseline_losses.json")
print("wrote", HERE / "baseline_loss_curve.png")
print("summary:", json.dumps(artifact["summary"], indent=2))
