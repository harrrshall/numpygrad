"""
layers.py - GPT-2 neural-network layers, built on the autograd Tensor.

Each layer is a small class that stores its parameters as `Tensor`s
(requires_grad=True) and whose __call__ runs the forward with tensor.py autograd
ops - so calling .backward() on any downstream scalar fills in every parameter's
.grad. The structure mirrors nanoGPT's model.py:

    Linear, LayerNorm, Embedding   - primitives
    MLP                            - c_fc -> GeLU -> c_proj
    MultiHeadAttention             - causal self-attention (nanoGPT CausalSelfAttention)
    TransformerBlock               - pre-LN: x + attn(ln_1(x)); x + mlp(ln_2(x))

Every layer exposes `parameters() -> {name: Tensor}` (for the optimizer and for
the layer-parity tests) and inherits `zero_grad()` from `Module`.

Convention: `Linear` stores its weight as **(in, out)** and computes `x @ W + b`
- the math convention. PyTorch's `nn.Linear` stores **(out, in)** and computes
`x @ W.T + b`; the parity test transposes when copying weights across.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tensor as T              # noqa: E402
from tensor import Tensor       # noqa: E402


def _prefixed(prefix, params):
    """Namespace a sub-module's parameter dict, e.g. {'W': ...} -> {'c_fc.W': ...}."""
    return {f"{prefix}.{k}": v for k, v in params.items()}


class Module:
    """Minimal base: a parameter dict and a graph-free way to clear gradients."""

    def parameters(self):
        return {}

    def zero_grad(self):
        for p in self.parameters().values():
            p.grad = None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class Linear(Module):
    """y = x @ W + b.  W is (n_in, n_out); b is (n_out,) or None."""

    def __init__(self, n_in, n_out, bias=True):
        self.W = Tensor(np.random.randn(n_in, n_out) * 0.02, requires_grad=True)
        self.b = Tensor(np.zeros(n_out), requires_grad=True) if bias else None

    def __call__(self, x):
        if self.b is None:
            return T.matmul(x, self.W)
        return T.linear(x, self.W, self.b)

    def parameters(self):
        return {'W': self.W} if self.b is None else {'W': self.W, 'b': self.b}


class LayerNorm(Module):
    """y = gamma * normalize(x) + beta over the last axis. eps = 1e-5 (nanoGPT).
    Biased variance (divisor D), matching torch.nn.LayerNorm and derivation.md item 11."""

    def __init__(self, n_dim, bias=True, eps=1e-5):
        self.gamma = Tensor(np.ones(n_dim), requires_grad=True)
        self.beta = Tensor(np.zeros(n_dim), requires_grad=True) if bias else None
        # ops.layernorm always takes a beta; with bias=False we feed a fixed zero
        # (requires_grad=False, so it contributes nothing and collects no grad).
        self._zero_beta = None if bias else Tensor(np.zeros(n_dim))
        self.eps = eps

    def __call__(self, x):
        beta = self.beta if self.beta is not None else self._zero_beta
        return T.layernorm(x, self.gamma, beta, self.eps)

    def parameters(self):
        if self.beta is None:
            return {'gamma': self.gamma}
        return {'gamma': self.gamma, 'beta': self.beta}


class Embedding(Module):
    """Lookup table, W is (n_vocab, n_dim). __call__(ids) gathers W[ids].
    Works for token embeddings (ids shape (B,T) -> (B,T,D)) and position
    embeddings (ids = arange(T), shape (T,) -> (T,D))."""

    def __init__(self, n_vocab, n_dim):
        # named `weight` (not `W`) to distinguish a lookup table from a Linear
        # weight: a Linear weight is stored transposed vs torch, a table is not.
        self.weight = Tensor(np.random.randn(n_vocab, n_dim) * 0.02, requires_grad=True)

    def __call__(self, ids):
        return T.embedding(self.weight, ids)

    def parameters(self):
        return {'weight': self.weight}


class MLP(Module):
    """nanoGPT MLP: c_fc (n_embd -> 4*n_embd) -> GeLU -> c_proj (4*n_embd -> n_embd)."""

    def __init__(self, n_embd, n_hidden=None, bias=True):
        n_hidden = 4 * n_embd if n_hidden is None else n_hidden
        self.c_fc = Linear(n_embd, n_hidden, bias=bias)
        self.c_proj = Linear(n_hidden, n_embd, bias=bias)

    def __call__(self, x):
        return self.c_proj(T.gelu(self.c_fc(x)))

    def parameters(self):
        return {**_prefixed('c_fc', self.c_fc.parameters()),
                **_prefixed('c_proj', self.c_proj.parameters())}


class MultiHeadAttention(Module):
    """Causal multi-head self-attention (nanoGPT CausalSelfAttention).

    Fused QKV projection `c_attn` (n_embd -> 3*n_embd), split into Q,K,V and into
    `n_head` heads; scaled dot-product attention with a causal mask; heads merged
    and run through the output projection `c_proj` (n_embd -> n_embd).
    """

    def __init__(self, n_embd, n_head, bias=True):
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.c_attn = Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = Linear(n_embd, n_embd, bias=bias)

    def __call__(self, x):
        B, Tn, D = x.shape
        H = self.n_head
        d = D // H

        qkv = self.c_attn(x)                                  # (B, T, 3D)
        Qf, Kf, Vf = T.split3(qkv)                            # each (B, T, D)

        def heads(z):                                         # (B,T,D) -> (B,H,T,d)
            return T.permute_heads(T.reshape(z, (B, Tn, H, d)))

        Q, K, V = heads(Qf), heads(Kf), heads(Vf)
        S = T.attention_scores(Q, K)                          # (B,H,T,T), already / sqrt(d)
        P = T.softmax(T.causal_mask(S))                       # causal mask -> softmax
        O = T.attention_output(P, V)                          # (B,H,T,d)
        O_merged = T.reshape(T.permute_heads(O), (B, Tn, D))  # (B,T,D)
        return self.c_proj(O_merged)

    def parameters(self):
        return {**_prefixed('c_attn', self.c_attn.parameters()),
                **_prefixed('c_proj', self.c_proj.parameters())}


class TransformerBlock(Module):
    """Pre-LN transformer block (nanoGPT Block):
        x = x + attn(ln_1(x))
        x = x + mlp(ln_2(x))
    """

    def __init__(self, n_embd, n_head, bias=True, eps=1e-5):
        self.ln_1 = LayerNorm(n_embd, bias=bias, eps=eps)
        self.attn = MultiHeadAttention(n_embd, n_head, bias=bias)
        self.ln_2 = LayerNorm(n_embd, bias=bias, eps=eps)
        self.mlp = MLP(n_embd, bias=bias)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def parameters(self):
        p = {}
        p.update(_prefixed('ln_1', self.ln_1.parameters()))
        p.update(_prefixed('attn', self.attn.parameters()))
        p.update(_prefixed('ln_2', self.ln_2.parameters()))
        p.update(_prefixed('mlp', self.mlp.parameters()))
        return p
