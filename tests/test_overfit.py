"""
test_overfit.py - Step 7 gate: the training loop actually learns.

The "overfit a tiny batch" sanity check. Take 16 sequences (4 batches of 4),
train ONLY on those for 500 steps, and confirm the loss collapses - the model
has far more than enough capacity to memorize 16 sequences, so a correct
training loop drives the loss to near zero. A plateau means the loop is broken:
usually the optimizer state isn't persisting between steps, or the model is
being rebuilt (resetting the weights), or gradients aren't being zeroed.

Gate: the loss reaches below 0.5 within 500 steps.

Setup:
  - A small GPT (vocab 128, block_size 32, 2 layers, 2 heads, n_embd 64) - tiny,
    so 500 steps run in a few seconds, but ~200k params, ample to memorize
    16 x 32 = 512 next-token predictions.
  - 16 fixed random token sequences, as 4 batches of 4, cycled deterministically.
  - AdamW with **weight_decay = 0**: weight decay regularizes *away* from
    memorizing, and this test is purely "does the loop learn". (Decoupled decay
    is already verified in Step 5 / Step 6.)
  - The model and optimizer are built ONCE; `trainer.train` reuses them every
    step, so the optimizer's m/v/t state persists - the thing under test.

Run:  .venv/bin/python tests/test_overfit.py
Writes: tests/overfit_results.md
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from gpt import GPT, GPTConfig        # noqa: E402
from optimizer import AdamW           # noqa: E402
from trainer import train             # noqa: E402

RESULTS_MD = os.path.join(_HERE, "overfit_results.md")

N_STEPS = 500
N_SEQ = 16
BATCH_SIZE = 4
LR = 3e-3
BETAS = (0.9, 0.99)
EPS = 1e-8
WEIGHT_DECAY = 0.0          # pure memorization test - no regularization
GRAD_CLIP = 1.0
GATE = 0.5


def main():
    np.random.seed(1337)

    cfg = GPTConfig(vocab_size=128, block_size=32, n_layer=2, n_head=2,
                    n_embd=64, bias=False)
    model = GPT(cfg)                                  # built ONCE

    # 16 fixed sequences = 4 batches of 4. Each sequence is block_size+1 tokens;
    # x = tokens[:-1], y = tokens[1:] (next-token targets).
    seqs = np.random.randint(0, cfg.vocab_size, size=(N_SEQ, cfg.block_size + 1))
    batches = []
    for b in range(N_SEQ // BATCH_SIZE):
        chunk = seqs[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
        batches.append((chunk[:, :-1].copy(), chunk[:, 1:].copy()))

    def get_batch(step):
        return batches[step % len(batches)]           # cycle through the 4 batches

    # optimizer built ONCE - its m/v/t state must persist across all 500 steps
    params = model.parameters()
    no_decay = {n for n, p in params.items() if p.data.ndim < 2}
    opt = AdamW(params, lr=LR, betas=BETAS, eps=EPS,
                weight_decay=WEIGHT_DECAY, no_decay=no_decay)

    print(f"overfit: {N_SEQ} sequences ({len(batches)} batches of {BATCH_SIZE}), "
          f"{N_STEPS} steps, lr={LR}, weight_decay={WEIGHT_DECAY}")
    print(f"  model: vocab={cfg.vocab_size} block_size={cfg.block_size} "
          f"n_layer={cfg.n_layer} n_head={cfg.n_head} n_embd={cfg.n_embd}")
    print("=" * 64)

    losses = train(model, opt, get_batch, N_STEPS, grad_clip=GRAD_CLIP, log_every=50)

    # final loss over ALL 16 sequences at once (the cleanest single number)
    _, full_loss = model(seqs[:, :-1], seqs[:, 1:])
    final_full = float(full_loss.data)

    init_loss = losses[0]
    min_loss = min(losses)
    last_loss = losses[-1]
    reached_below = min_loss < GATE
    gate_ok = final_full < GATE and reached_below

    print("=" * 64)
    print(f"  initial loss (step 0)        : {init_loss:.4f}")
    print(f"  min loss over {N_STEPS} steps      : {min_loss:.4f}")
    print(f"  loss at final step           : {last_loss:.4f}")
    print(f"  loss over all {N_SEQ} sequences   : {final_full:.4f}")
    print(f"  {'OVERFIT OK - the training loop learns' if gate_ok else 'FAILED - loss did not reach below 0.5 (loop bug?)'}")

    write_md(cfg, losses, init_loss, min_loss, last_loss, final_full, gate_ok)
    print(f"  wrote {RESULTS_MD}")
    return 0 if gate_ok else 1


def write_md(cfg, losses, init_loss, min_loss, last_loss, final_full, gate_ok):
    L = []
    L.append("# Overfit Sanity-Check Results")
    L.append("")
    L.append('The "overfit a tiny batch" check: confirm the training loop actually learns. '
             "16 sequences (4 batches of 4) are trained on for 500 steps; the model has far "
             "more capacity than needed to memorize them, so a correct loop drives the loss "
             "to near zero. A plateau would mean a loop bug - optimizer state not persisting, "
             "the model being rebuilt, or gradients not zeroed.")
    L.append("")
    L.append("## Setup")
    L.append("")
    L.append(f"- Model: small GPT - `vocab_size={cfg.vocab_size}`, `block_size={cfg.block_size}`, "
             f"`n_layer={cfg.n_layer}`, `n_head={cfg.n_head}`, `n_embd={cfg.n_embd}`, "
             f"`bias=False` (~200k params - tiny, but ample to memorize 16x32 = 512 "
             "next-token predictions).")
    L.append(f"- Data: {N_SEQ} fixed random token sequences, as {N_SEQ // BATCH_SIZE} batches "
             f"of {BATCH_SIZE}, cycled deterministically (`step % {N_SEQ // BATCH_SIZE}`).")
    L.append(f"- Optimizer: `AdamW(lr={LR}, betas={BETAS}, eps={EPS}, "
             f"weight_decay={WEIGHT_DECAY})`, `grad_clip={GRAD_CLIP}`. Built **once** - its "
             "m/v/t state persists across all 500 steps.")
    L.append(f"- `weight_decay = {WEIGHT_DECAY}`: weight decay regularizes *away* from "
             "memorizing, so a pure overfit test turns it off. (Decoupled decay is already "
             "verified in Steps 5-6.)")
    L.append(f"- **Gate:** loss reaches below `{GATE}` within {N_STEPS} steps.")
    L.append("")
    L.append("## Loss curve")
    L.append("")
    L.append("| Step | Loss |")
    L.append("|---|---|")
    marks = list(range(0, len(losses), 50))
    if (len(losses) - 1) not in marks:
        marks.append(len(losses) - 1)
    for s in marks:
        L.append(f"| {s} | {losses[s]:.4f} |")
    L.append("")
    L.append(f"- Initial loss (step 0): `{init_loss:.4f}`  (~ln(vocab) = "
             f"`{np.log(cfg.vocab_size):.4f}`, i.e. uniform random)")
    L.append(f"- Min loss over {N_STEPS} steps: `{min_loss:.4f}`")
    L.append(f"- Loss at the final step: `{last_loss:.4f}`")
    L.append(f"- Loss over all {N_SEQ} sequences at once (final): `{final_full:.4f}`")
    L.append("")
    L.append("## Result")
    L.append("")
    if gate_ok:
        L.append(f"**Overfit OK.** The loss collapsed from `{init_loss:.2f}` to "
                 f"`{final_full:.4f}` - well below the `{GATE}` gate. The training loop "
                 "learns: the optimizer state persists across steps, the autograd graph is "
                 "rebuilt correctly each step over the same parameters, and gradients are "
                 "zeroed between steps. Ready for real training.")
    else:
        L.append(f"**FAILED.** The loss did not reach below `{GATE}` (min `{min_loss:.4f}`, "
                 f"final over all sequences `{final_full:.4f}`). A model this size memorizes "
                 "16 sequences trivially, so a plateau points at the loop: optimizer state "
                 "not persisting between steps, the model being rebuilt, or gradients not "
                 "being zeroed.")
    L.append("")
    L.append("## How to run")
    L.append("")
    L.append("```")
    L.append(".venv/bin/python tests/test_overfit.py")
    L.append("```")
    L.append("")
    with open(RESULTS_MD, "w") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
