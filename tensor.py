"""
tensor.py - A minimal reverse-mode autograd Tensor.

A `Tensor` wraps a NumPy array and records the op that produced it. Calling
`.backward()` on a scalar walks the recorded graph in reverse topological order,
calling each op's `_backward` closure to push gradients to its parents.

Design rule that makes this trustworthy: every op is a thin wrapper around a
verified `ops.py` closure (the same closures finite-difference checked in
tests/gradcheck.py). The op's `_backward` calls that *same* closure - so a
gradient computed through the Tensor graph is bit-identical to the hand-rolled
`backward_fn` result from Step 1, not merely close to it.

The one piece of genuinely new machinery is **gradient accumulation**: when a
tensor feeds into more than one op (a residual connection, weight tying, or
literally `x * x`), each consumer pushes a gradient and they must *add*. That
is `Tensor._accum`. Overwrite instead of accumulate and `y = x + x` silently
gives `dx = 1` instead of `2`; the smoke test at the bottom guards exactly this.
"""
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests"))
import ops  # noqa: E402  (the verified Step 1 forward/backward closures)

# Global autograd-recording switch. Inside a `with no_grad():` block, ops skip
# building the graph entirely - just compute the forward and return a plain
# Tensor. Used for evaluation (perplexity, benchmarks): no graph means no
# captured intermediates, so a big model's forward stays fast and low-memory.
_NO_GRAD = False


class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = None                       # dL/d(self), same shape as data
        self.requires_grad = bool(requires_grad)
        # _backward(grad): given this tensor's gradient, push contributions to
        # its parents. It takes `grad` as an ARGUMENT and never closes over the
        # tensor it belongs to - otherwise `t` <-> `t._backward` would be a
        # reference cycle, the graph could only be freed by the cyclic GC, and
        # memory would balloon across a training run (one ~700MB graph per step
        # for a 50k-vocab model). Argument-passing keeps the graph a pure DAG
        # that reference-counting frees the instant the loss is dropped.
        self._backward = lambda grad: None     # set by the op that produced this tensor
        self._parents = ()                     # tensors this one was computed from

    # -- core -------------------------------------------------------------

    def backward(self, grad=None):
        """Reverse-mode backward pass over self and all its ancestors.

        With no argument this is only valid on a scalar output (the loss),
        seeded with dL/dL = 1. An explicit `grad` (same shape as data) may be
        passed for non-scalar outputs.
        """
        # topological order: a node appears after all of its parents
        topo, visited = [], set()

        def build(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            for p in t._parents:
                build(p)
            if t.requires_grad:                # only schedule nodes that need a grad
                topo.append(t)

        build(self)

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "backward() without an explicit gradient is only valid on a "
                    f"scalar output; this tensor has shape {self.data.shape}")
            self.grad = np.ones_like(self.data)
        else:
            self.grad = np.asarray(grad, dtype=np.float64)

        for t in reversed(topo):               # children before parents
            t._backward(t.grad)                # pass the grad in - no cycle (see __init__)

    def zero_grad(self):
        """Clear `.grad` on self and every ancestor, ready for a fresh backward."""
        visited = set()

        def clear(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            t.grad = None
            for p in t._parents:
                clear(p)

        clear(self)

    def _accum(self, grad):
        """Accumulate `grad` into self.grad. THE rule that keeps multi-use
        tensors correct: a tensor used in k places receives k contributions
        and they sum. First contribution initialises; the rest add."""
        if not self.requires_grad:
            return
        if self.grad is None:
            self.grad = np.array(grad, dtype=np.float64)   # copy on first write
        else:
            self.grad = self.grad + grad                   # accumulate

    # -- operator overloads ----------------------------------------------

    def __add__(self, other):  return _add(self, _as_tensor(other))
    def __radd__(self, other): return _add(_as_tensor(other), self)
    def __mul__(self, other):  return _mul(self, _as_tensor(other))
    def __rmul__(self, other): return _mul(_as_tensor(other), self)
    def __neg__(self):         return _mul(self, Tensor(-1.0))
    def __sub__(self, other):  return _add(self, -_as_tensor(other))
    def __rsub__(self, other): return _add(_as_tensor(other), -self)
    def __matmul__(self, other): return matmul(self, _as_tensor(other))

    def sum(self):
        return _sum(self)

    # -- conveniences -----------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return (f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad}, "
                f"grad={'set' if self.grad is not None else 'None'})")


def _as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor(x)


class no_grad:
    """Context manager that disables autograd recording (PyTorch-style).

    Inside `with no_grad():`, every op computes its forward and returns a plain
    Tensor with no `_parents` / `_backward` - no graph is built and no
    intermediate arrays are captured. For evaluation only (the result cannot be
    backpropagated)."""

    def __enter__(self):
        global _NO_GRAD
        self._prev = _NO_GRAD
        _NO_GRAD = True
        return self

    def __exit__(self, *exc):
        global _NO_GRAD
        _NO_GRAD = self._prev
        return False


# --------------------------------------------------------------------------
# the wrapper that turns a verified ops.py closure into a graph-recording op
# --------------------------------------------------------------------------

def _apply(fn, tensor_inputs, const_inputs=None):
    """Build a Tensor op from a verified ops.py function.

    fn            : an ops.py function returning (output, backward_fn)
    tensor_inputs : {ops_param_name: Tensor} - the differentiable inputs. The
                    grad-dict key returned by `backward_fn` equals the param
                    name, so the same dict drives both forward and backward.
                    If one Tensor is passed under two names (e.g. mul(x, x)),
                    `.items()` visits it twice and `_accum` adds both - which
                    is exactly the gradient-accumulation behaviour we want.
    const_inputs  : {ops_param_name: value} - non-differentiable args (ints,
                    integer id arrays, eps, ...).
    """
    const_inputs = const_inputs or {}
    raw = {name: t.data for name, t in tensor_inputs.items()}
    out_data, bwd = fn(**raw, **const_inputs)

    if _NO_GRAD:                                   # eval mode: skip the graph entirely
        return Tensor(out_data, requires_grad=False)

    out = Tensor(out_data, requires_grad=any(t.requires_grad
                                             for t in tensor_inputs.values()))
    out._parents = tuple(tensor_inputs.values())

    def _backward(out_grad):                   # takes the grad as an arg - does NOT
        if out_grad is None:                   # close over `out` (avoids a ref cycle)
            return
        grads = bwd(out_grad)                  # the SAME closure Step 1 checked
        for name, t in tensor_inputs.items():
            t._accum(grads[name])

    out._backward = _backward
    return out


# the three ops the operator overloads need (the rest land in Piece 2)
def _add(a, b):
    return _apply(ops.add, {'a': a, 'b': b})

def _mul(a, b):
    return _apply(ops.mul, {'a': a, 'b': b})

def _sum(x):
    return _apply(ops.sum_all, {'x': x})


# --------------------------------------------------------------------------
# the op library - one thin wrapper per verified ops.py primitive.
# Every _backward calls the exact closure finite-difference checked in Step 1,
# so a gradient through the Tensor graph is bit-identical to ops.py's backward.
# --------------------------------------------------------------------------

def _raw(x):
    """Unwrap a Tensor to its ndarray; pass non-Tensors (ints, id arrays) through."""
    return x.data if isinstance(x, Tensor) else x

def scale(x, c):
    return _apply(ops.scale, {'x': x}, {'c': c})

def relu(x):
    return _apply(ops.relu, {'x': x})

def gelu(x):
    return _apply(ops.gelu, {'x': x})

def matmul(x, w):
    return _apply(ops.matmul, {'x': x, 'W': w})

def linear(x, w, b):
    return _apply(ops.linear, {'x': x, 'W': w, 'b': b})

def sum_lastaxis(x):
    return _apply(ops.sum_lastaxis, {'x': x})

def mean_lastaxis(x):
    return _apply(ops.mean_lastaxis, {'x': x})

def transpose(x):
    return _apply(ops.transpose2d, {'x': x})

def permute_heads(x):
    """(B,T,H,d) <-> (B,H,T,d), the attention head-split permutation."""
    return _apply(ops.permute_bhtd, {'x': x})

def reshape(x, newshape):
    return _apply(ops.reshape_op, {'x': x}, {'newshape': newshape})

def slice_lastaxis(x, start, stop):
    return _apply(ops.slice_lastaxis, {'x': x}, {'start': start, 'stop': stop})

def split3(x):
    """Split a Tensor along the last axis into 3 equal parts (fused QKV -> Q,K,V).
    Just three slices: each scatters its gradient back into the parent, whose
    _accum sums the three disjoint slices into the full gradient."""
    d = x.shape[-1] // 3
    return (slice_lastaxis(x, 0, d),
            slice_lastaxis(x, d, 2 * d),
            slice_lastaxis(x, 2 * d, 3 * d))

def softmax(x):
    return _apply(ops.softmax_op, {'x': x})

def cross_entropy(p, y):
    return _apply(ops.cross_entropy_from_probs, {'p': p}, {'y': _raw(y)})

def softmax_cross_entropy(logits, targets):
    return _apply(ops.softmax_ce_fused, {'logits': logits}, {'targets': _raw(targets)})

def layernorm(x, gamma, beta, eps=1e-5):
    return _apply(ops.layernorm, {'x': x, 'gamma': gamma, 'beta': beta}, {'eps': eps})

def embedding(w_e, ids):
    return _apply(ops.embedding, {'W_e': w_e}, {'ids': _raw(ids)})

def positional_embedding(w_p, T, B):
    return _apply(ops.positional_embedding, {'W_p': w_p}, {'T': T, 'B': B})

def lm_head(h, w_e):
    return _apply(ops.lm_head, {'h': h, 'W_e': w_e})

def attention_scores(q, k):
    return _apply(ops.attention_scores, {'Q': q, 'K': k})

def causal_mask(s, fill=-1e9):
    return _apply(ops.causal_mask, {'S': s}, {'fill': fill})

def attention_output(p, v):
    return _apply(ops.attention_output, {'P': p, 'V': v})


# --------------------------------------------------------------------------
# smoke test - the gradient-accumulation trap
# --------------------------------------------------------------------------

def _smoke_test():
    print("tensor.py smoke test - gradient accumulation")
    print("=" * 60)
    rng = np.random.RandomState(0)
    ok = True

    def check(name, got, want):
        nonlocal ok
        exact = np.array_equal(got, want)
        ok &= exact
        print(f"  {'PASS' if exact else 'FAIL'}  {name:<34} "
              f"max|diff|={np.abs(got - want).max():.1e}")

    # y = x * x  ->  dx = 2x   (x used twice in one op)
    x = Tensor(rng.randn(3, 4), requires_grad=True)
    (x * x).sum().backward()
    check("y = x*x      -> dx = 2x", x.grad, 2.0 * x.data)

    # y = x + x  ->  dx = 2     (the residual-halving trap)
    x = Tensor(rng.randn(3, 4), requires_grad=True)
    (x + x).sum().backward()
    check("y = x+x      -> dx = 2", x.grad, np.full((3, 4), 2.0))

    # diamond: a = 2x, b = 3x, y = a + b  ->  dx = 5   (x used in two ops)
    x = Tensor(rng.randn(3, 4), requires_grad=True)
    a, b = x * 2.0, x * 3.0
    (a + b).sum().backward()
    check("diamond a+b  -> dx = 5", x.grad, np.full((3, 4), 5.0))

    # x^3 via x*x*x  ->  dx = 3x^2
    x = Tensor(rng.randn(3, 4), requires_grad=True)
    (x * x * x).sum().backward()
    check("y = x*x*x    -> dx = 3x^2", x.grad, 3.0 * x.data ** 2)

    # zero_grad clears, and a second backward reproduces the first exactly
    x = Tensor(rng.randn(3, 4), requires_grad=True)
    loss = (x * x).sum()
    loss.backward()
    g1 = x.grad.copy()
    loss.zero_grad()
    cleared = (x.grad is None)
    ok &= cleared
    print(f"  {'PASS' if cleared else 'FAIL'}  {'zero_grad clears grad':<34} "
          f"grad is None: {cleared}")
    (x * x).sum().backward()
    check("re-run backward reproduces", x.grad, g1)

    print("=" * 60)
    print(f"  {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_smoke_test())
