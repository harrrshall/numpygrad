"""
trainer.py - the training loop and gradient clipping.

`train(model, optimizer, get_batch, n_steps, ...)` runs the standard loop:

    for each step:  zero_grad -> forward -> backward -> [grad clip] -> optimizer.step

Two things make it actually learn, and getting either wrong makes the loss
plateau (see tests/test_overfit.py):

  - The model and optimizer are built ONCE by the caller and reused every step.
    The optimizer's moment state (m, v, t) therefore persists across steps - a
    fresh optimizer per step would reset the EMAs and the bias correction.
  - Each forward rebuilds a fresh autograd graph, but over the SAME parameter
    Tensor objects. Rebuilding the *model* per step would reset the weights;
    not zero-ing grads would let them accumulate across steps.

`clip_grad_norm` matches `torch.nn.utils.clip_grad_norm_`: scale every gradient
so the global L2 norm is at most `max_norm`. nanoGPT uses `grad_clip=1.0`.

`get_lr` is nanoGPT's cosine-with-warmup learning-rate schedule.
"""
import math

import numpy as np


def get_lr(it, learning_rate, min_lr, warmup_iters, lr_decay_iters):
    """nanoGPT's learning-rate schedule: linear warmup for `warmup_iters` steps,
    then cosine decay from `learning_rate` down to `min_lr` by `lr_decay_iters`,
    then flat at `min_lr`. Identical to `get_lr` in nanogpt/train.py."""
    if it < warmup_iters:                                  # 1) linear warmup
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:                                # 2) past decay -> floor
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)   # 3) cosine
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0
    return min_lr + coeff * (learning_rate - min_lr)


def clip_grad_norm(params, max_norm):
    """Scale gradients in place so the global L2 norm is <= max_norm (matches
    torch.nn.utils.clip_grad_norm_). `params` is a {name: Tensor} dict. Returns
    the total gradient norm BEFORE clipping."""
    grads = [p.grad for p in params.values() if p.grad is not None]
    total_norm = float(np.sqrt(sum(np.sum(g * g) for g in grads)))
    clip_coef = max_norm / (total_norm + 1e-6)        # PyTorch's +1e-6 in the denom
    if clip_coef < 1.0:                               # only scale down, never up
        for g in grads:
            g *= clip_coef
    return total_norm


def train(model, optimizer, get_batch, n_steps, grad_clip=None, lr_schedule=None,
          log_every=0):
    """Run `n_steps` of training.

    model       : a GPT (built once, reused every step)
    optimizer   : an AdamW (built once - its m/v/t state persists across steps)
    get_batch   : get_batch(step) -> (x, y) integer arrays
    grad_clip   : if set, clip the global grad norm to this before each step
    lr_schedule : if set, lr_schedule(step) -> lr; written to optimizer.lr each step
    log_every   : if > 0, print the loss every `log_every` steps

    Returns the list of per-step losses.
    """
    params = model.parameters()                       # same Tensor objects every step
    losses = []
    for step in range(n_steps):
        if lr_schedule is not None:
            optimizer.lr = lr_schedule(step)          # set this step's learning rate
        x, y = get_batch(step)
        optimizer.zero_grad()                         # clear the previous step's grads
        _, loss = model(x, y)                         # fresh graph over the same params
        loss.backward()
        if grad_clip is not None:
            clip_grad_norm(params, grad_clip)
        optimizer.step()                              # in-place update; m/v/t carry over
        losses.append(float(loss.data))
        if log_every and (step % log_every == 0 or step == n_steps - 1):
            print(f"  step {step:4d}  loss {losses[-1]:.4f}")
    return losses
