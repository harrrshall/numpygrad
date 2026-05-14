"""
optimizer.py - AdamW, matching torch.optim.AdamW numerically.

AdamW = Adam with **decoupled** weight decay (Loshchilov & Hutter). The decay
term is applied directly to the parameter - `p *= (1 - lr*weight_decay)` - and
is NOT folded into the gradient. That decoupling is the whole point of AdamW and
the thing implementations get wrong; see derivation.md item 24.

One step, matching PyTorch's torch.optim.AdamW:

    1. p     <- p * (1 - lr * weight_decay)              # decoupled weight decay
    2. m     <- beta1*m + (1-beta1)*g                    # 1st-moment EMA
       v     <- beta2*v + (1-beta2)*g^2                  # 2nd-moment EMA
    3. m_hat <- m / (1 - beta1^t),  v_hat <- v / (1 - beta2^t)   # bias correction
       p     <- p - lr * m_hat / (sqrt(v_hat) + eps)             # Adam step

`m` and `v` start at zero, so step t's bias correction divides by `1 - beta^t`
to undo the cold-start bias. Verified against torch.optim.AdamW for 10 steps in
tests/test_optimizer_parity.py.

Parameter groups: nanoGPT applies weight decay only to >=2D tensors (matmul /
embedding weights), with 1D tensors (biases, LayerNorm) excluded. This `AdamW`
takes a scalar `weight_decay` plus an optional `no_decay` set of parameter names
that use weight_decay=0 - so the caller reproduces nanoGPT's grouping with
`no_decay = {n for n, p in params.items() if p.data.ndim < 2}`.
"""
import numpy as np


class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1,
                 no_decay=()):
        """params:   dict {name: Tensor} - e.g. model.parameters().
        no_decay: names of parameters that use weight_decay = 0. nanoGPT excludes
                  every 1D tensor (biases, LayerNorm) from weight decay; the
                  caller passes those names here. Empty by default (uniform
                  weight decay over all parameters)."""
        self.params = dict(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.no_decay = set(no_decay)
        self.t = 0
        # per-parameter optimizer state (1st/2nd moment EMAs), keyed by name,
        # initialised to zero - the bias correction below accounts for this.
        self.m = {name: np.zeros_like(p.data) for name, p in self.params.items()}
        self.v = {name: np.zeros_like(p.data) for name, p in self.params.items()}

    def step(self):
        """One AdamW update, in place, using each parameter's current `.grad`."""
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t          # bias-correction denominators
        bc2 = 1.0 - self.beta2 ** self.t
        for name, p in self.params.items():
            if p.grad is None:
                continue
            g = p.grad
            wd = 0.0 if name in self.no_decay else self.weight_decay

            # 1. decoupled weight decay - applied to the parameter, NOT the
            #    gradient. This is the AdamW step; with weight_decay=0 it is a
            #    no-op (x * 1.0 == x) and AdamW reduces to plain Adam.
            p.data *= (1.0 - self.lr * wd)

            # 2. moment EMAs
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (g * g)

            # 3. bias-corrected Adam step
            m_hat = self.m[name] / bc1
            v_hat = self.v[name] / bc2
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Clear every parameter's gradient (call before the next backward)."""
        for p in self.params.values():
            p.grad = None
