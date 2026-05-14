"""
ops.py - Standalone NumPy forward/backward implementations of every gradient
formula derived in ../derivation.md.

Each op has the signature:

    forward(*inputs) -> (output, backward_fn)

where `backward_fn(upstream_grad) -> {param_name: grad, ...}`. The dict keys are
exactly the *differentiable* parameter names of `forward`, and every grad has the
same shape as its corresponding input (the convention from derivation.md item 1).

These are the verification prototypes the autograd graph will later be built
from. Every formula here is finite-difference checked by gradcheck.py.

All math is float64. References (item N) point at derivation.md.
"""
import numpy as np
from scipy.special import erf as _erf   # NumPy has no erf; scipy.special does

# GeLU: exact (erf) form, matching nanoGPT's nn.GELU() -- derivation.md item 4b
_SQRT2 = np.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


# --------------------------------------------------------------------------
# Tier 1: warm-ups
# --------------------------------------------------------------------------

def _unbroadcast(grad, shape):
    """Reduce `grad` back to `shape`, reversing NumPy broadcasting: sum away any
    leading axes broadcasting added, and any axis where `shape` is 1 but `grad`
    is larger. The backward of every broadcasting op (add / mul) needs this."""
    grad = np.asarray(grad, dtype=np.float64)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def add(a, b):
    """Item 2. Y = A + B, with full NumPy broadcasting.
    dA = unbroadcast(dY, A.shape), dB = unbroadcast(dY, B.shape). For equal
    shapes this is exactly dA = dB = dY (the original item-2 result)."""
    Y = a + b
    def bwd(dY):
        return {'a': _unbroadcast(dY, np.shape(a)).copy(),
                'b': _unbroadcast(dY, np.shape(b)).copy()}
    return Y, bwd


def mul(a, b):
    """Element-wise product Y = A (.) B, with full NumPy broadcasting - the
    two-tensor generalisation of the item-3 scalar scale, and the building
    block of every (.) in the derivations.
    dA = unbroadcast(dY (.) B, A.shape), dB = unbroadcast(dY (.) A, B.shape)."""
    Y = a * b
    def bwd(dY):
        return {'a': _unbroadcast(dY * b, np.shape(a)),
                'b': _unbroadcast(dY * a, np.shape(b))}
    return Y, bwd


def add_bias(x, b):
    """Item 2 (broadcast case). Y = X + b, b broadcast over rows.
    dX = dY ; db = dY summed over the broadcast axes."""
    Y = x + b
    def bwd(dY):
        # sum over every axis of dY except the last (which matches b)
        axes = tuple(range(dY.ndim - 1))
        return {'x': dY.copy(), 'b': dY.sum(axis=axes)}
    return Y, bwd


def scale(x, c):
    """Item 3. Y = c * X (c a constant). dX = c * dY."""
    Y = c * x
    def bwd(dY):
        return {'x': c * dY}
    return Y, bwd


def relu(x):
    """Item 4a. Y = max(0, X). dX = 1[X > 0] (.) dY."""
    Y = np.maximum(0.0, x)
    mask = (x > 0.0)
    def bwd(dY):
        return {'x': mask * dY}
    return Y, bwd


def gelu(x):
    """Item 4b. GeLU, exact form -- matches nanoGPT's nn.GELU().
    f(x)  = x * Phi(x),  Phi(x) = 0.5 (1 + erf(x / sqrt(2)))   [standard normal CDF]
    f'(x) = Phi(x) + x * phi(x),  phi(x) = exp(-x^2/2)/sqrt(2 pi)  [standard normal PDF]"""
    Phi = 0.5 * (1.0 + _erf(x / _SQRT2))
    Y = x * Phi
    def bwd(dY):
        phi = np.exp(-0.5 * x * x) * _INV_SQRT_2PI
        fprime = Phi + x * phi
        return {'x': fprime * dY}
    return Y, bwd


# --------------------------------------------------------------------------
# Tier 2: linear algebra ops
# --------------------------------------------------------------------------

def matmul(x, W):
    """Item 5. Y = X W (no bias). x may be (..., K) - the leading axes are
    flattened for the dW contraction (X^T dY is only the 2-D special case).
    dX = dY W^T,  dW = X_flat^T dY_flat."""
    Y = x @ W
    K, N = W.shape
    def bwd(dY):
        dx = dY @ W.T
        dW = x.reshape(-1, K).T @ dY.reshape(-1, N)
        return {'x': dx, 'W': dW}
    return Y, bwd


def linear(x, W, b):
    """Item 5 (affine). Y = X W + b.  Supports x of shape (..., K): the leading
    axes are flattened for the dW contraction.
    dX = dY W^T, dW = X_flat^T dY_flat, db = dY summed over the leading axes."""
    Y = x @ W + b
    K = W.shape[0]
    N = W.shape[1]
    def bwd(dY):
        dx = dY @ W.T
        dW = x.reshape(-1, K).T @ dY.reshape(-1, N)
        db = dY.reshape(-1, N).sum(axis=0)
        return {'x': dx, 'W': dW, 'b': db}
    return Y, bwd


def sum_lastaxis(x):
    """Item 6. Y = X.sum(axis=-1). dX = broadcast(dY) over the summed axis."""
    Y = x.sum(axis=-1)
    def bwd(dY):
        return {'x': np.broadcast_to(dY[..., None], x.shape).copy()}
    return Y, bwd


def mean_lastaxis(x):
    """Item 6. Y = X.mean(axis=-1). dX = broadcast(dY) / N."""
    N = x.shape[-1]
    Y = x.mean(axis=-1)
    def bwd(dY):
        return {'x': np.broadcast_to(dY[..., None], x.shape).copy() / N}
    return Y, bwd


def sum_all(x):
    """Item 6 (full reduction). Y = sum of all elements of X (a scalar). Every
    element contributes with weight 1, so dX = dY broadcast over X.shape. Used
    to form the scalar loss L = sum(output (.) upstream) in the autograd tests."""
    Y = np.asarray(x, dtype=np.float64).sum()
    def bwd(dY):
        return {'x': np.full(np.shape(x), dY, dtype=np.float64)}
    return Y, bwd


def transpose2d(x):
    """Item 7. Y = X^T. dX = dY^T."""
    Y = x.T
    def bwd(dY):
        return {'x': dY.T}
    return Y, bwd


def permute_bhtd(x):
    """Item 7 (multi-axis). Y = transpose(X, (0,2,1,3)) - the attention
    head-split permutation. It is its own inverse, so dX = transpose(dY, (0,2,1,3))."""
    Y = x.transpose(0, 2, 1, 3)
    def bwd(dY):
        return {'x': dY.transpose(0, 2, 1, 3)}
    return Y, bwd


def reshape_op(x, newshape):
    """Item 7. Y = reshape(X, newshape). dX = reshape(dY, X.shape)."""
    Y = x.reshape(newshape)
    def bwd(dY):
        return {'x': dY.reshape(x.shape)}
    return Y, bwd


def slice_lastaxis(x, start, stop):
    """Item 7 (slice). Y = X[..., start:stop]. The sliced-away positions of X
    do not influence Y, so backward scatters dY into a zero array at exactly
    that slice. Used to split the fused QKV projection into Q, K, V."""
    Y = x[..., start:stop].copy()
    def bwd(dY):
        dx = np.zeros_like(x)
        dx[..., start:stop] = dY
        return {'x': dx}
    return Y, bwd


# --------------------------------------------------------------------------
# Tier 3: loss layer
# --------------------------------------------------------------------------

def softmax_op(x):
    """Item 8. p = softmax(x) over the last axis (numerically stable).
    dx = p (.) (dp - (dp . p)), the per-row dot taken over the last axis."""
    z = x - x.max(axis=-1, keepdims=True)
    with np.errstate(under='ignore'):
        e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)
    def bwd(dp):
        dx = p * (dp - np.sum(dp * p, axis=-1, keepdims=True))
        return {'x': dx}
    return p, bwd


def cross_entropy_from_probs(p, y):
    """Item 9. L = -sum_i y_i log(p_i)  (summed over all elements).
    dp = -y / p.  y (the target) is treated as a constant."""
    L = -np.sum(y * np.log(p))
    def bwd(dL):
        return {'p': (-y / p) * dL}
    return L, bwd


def softmax_ce_fused(logits, targets):
    """Item 10. Fused softmax + cross-entropy, mean-reduced over all B*T tokens.
    L = mean_n CE(softmax(logits_n), onehot(targets_n)).
    dlogits = (p - onehot) / N,  N = B*T."""
    B, T, V = logits.shape
    N = B * T
    flat = logits.reshape(N, V)
    tgt = targets.reshape(N)
    z = flat - flat.max(axis=-1, keepdims=True)
    with np.errstate(under='ignore'):
        e = np.exp(z)
    p = e / e.sum(axis=-1, keepdims=True)              # (N, V)
    rows = np.arange(N)
    L = -np.mean(np.log(p[rows, tgt]))
    def bwd(dL):
        dflat = p.copy()
        dflat[rows, tgt] -= 1.0
        dflat = dflat / N * dL
        return {'logits': dflat.reshape(B, T, V)}
    return L, bwd


# --------------------------------------------------------------------------
# Tier 4: normalization
# --------------------------------------------------------------------------

def layernorm(x, gamma, beta, eps):
    """Item 11. y = gamma (.) xhat + beta,  xhat = (x - mu) / sqrt(var + eps),
    mu/var over the last axis (size D, biased variance).
        dxhat  = gamma (.) dy
        dbeta  = sum over leading axes of dy
        dgamma = sum over leading axes of (dy (.) xhat)
        dx = 1/(D s) * (D dxhat - sum_j dxhat_j - xhat sum_j dxhat_j xhat_j)
    """
    D = x.shape[-1]
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = np.mean(xc * xc, axis=-1, keepdims=True)
    s = np.sqrt(var + eps)
    xhat = xc / s
    y = gamma * xhat + beta
    lead = tuple(range(x.ndim - 1))
    def bwd(dy):
        dgamma = np.sum(dy * xhat, axis=lead)
        dbeta = np.sum(dy, axis=lead)
        dxhat = gamma * dy
        sum_dxhat = np.sum(dxhat, axis=-1, keepdims=True)
        sum_dxhat_xhat = np.sum(dxhat * xhat, axis=-1, keepdims=True)
        dx = (1.0 / (D * s)) * (D * dxhat - sum_dxhat - xhat * sum_dxhat_xhat)
        return {'x': dx, 'gamma': dgamma, 'beta': dbeta}
    return y, bwd


# --------------------------------------------------------------------------
# Tier 5: embeddings and weight tying
# --------------------------------------------------------------------------

def embedding(W_e, ids):
    """Item 12. Y[b,t] = W_e[ids[b,t]]. Backward is a scatter-add; np.add.at
    is REQUIRED so that repeated ids accumulate instead of being overwritten."""
    Y = W_e[ids]
    def bwd(dY):
        dW_e = np.zeros_like(W_e)
        np.add.at(dW_e, ids, dY)
        return {'W_e': dW_e}
    return Y, bwd


def embedding_buggy(W_e, ids):
    """Item 12 - NEGATIVE CONTROL. Identical forward, but the backward uses
    `dW_e[ids] += dY`, which silently drops gradient when an id repeats.
    This op is EXPECTED to fail the gradient check."""
    Y = W_e[ids]
    def bwd(dY):
        dW_e = np.zeros_like(W_e)
        dW_e[ids] += dY               # BUG: no accumulation on repeated ids
        return {'W_e': dW_e}
    return Y, bwd


def positional_embedding(W_p, T, B):
    """Item 13. Y[b,t] = W_p[t] for t < T, broadcast over the batch.
    dW_p[:T] = dY.sum(axis=0); rows >= T stay zero."""
    Y = np.broadcast_to(W_p[:T], (B, T, W_p.shape[1])).copy()
    def bwd(dY):
        dW_p = np.zeros_like(W_p)
        dW_p[:T] = dY.sum(axis=0)
        return {'W_p': dW_p}
    return Y, bwd


def lm_head(h, W_e):
    """Item 14 (Part A). logits = h @ W_e^T  (weight-tied output projection).
    dh = dlogits @ W_e ; dW_e = dlogits_flat^T @ h_flat  (the head contribution
    only; the embedding-lookup contribution accumulates separately - see
    tiny_gpt)."""
    logits = h @ W_e.T
    D = h.shape[-1]
    V = W_e.shape[0]
    def bwd(dlogits):
        dh = dlogits @ W_e
        dW_e = dlogits.reshape(-1, V).T @ h.reshape(-1, D)
        return {'h': dh, 'W_e': dW_e}
    return logits, bwd


# --------------------------------------------------------------------------
# Tier 6: attention
# --------------------------------------------------------------------------

def attention_scores(Q, K):
    """Item 16. S = Q K^T / sqrt(d), Q,K (B,H,T,d).
    dS' = dS / sqrt(d) ; dQ = dS' @ K ; dK = dS'^T @ Q  (transpose last two axes)."""
    d = Q.shape[-1]
    scale_c = 1.0 / np.sqrt(d)
    S = (Q @ K.transpose(0, 1, 3, 2)) * scale_c
    def bwd(dS):
        dSp = dS * scale_c
        dQ = dSp @ K
        dK = dSp.transpose(0, 1, 3, 2) @ Q
        return {'Q': dQ, 'K': dK}
    return S, bwd


def causal_mask(S, fill):
    """Item 17. S_masked = S + M, M[t,tau] = fill for tau > t (future), else 0.
    Backward is the identity: dS = dS_masked."""
    T = S.shape[-1]
    M = np.triu(np.full((T, T), fill, dtype=np.float64), k=1)
    S_masked = S + M
    def bwd(dS_masked):
        return {'S': dS_masked.copy()}
    return S_masked, bwd


def attention_output(P, V):
    """Item 19. O = P V, P (B,H,T,T), V (B,H,T,d).
    dP = dO @ V^T ; dV = P^T @ dO  (transpose last two axes)."""
    O = P @ V
    def bwd(dO):
        dP = dO @ V.transpose(0, 1, 3, 2)
        dV = P.transpose(0, 1, 3, 2) @ dO
        return {'P': dP, 'V': dV}
    return O, bwd


def sdpa_causal(Q, K, V):
    """Item 20. Scaled dot-product causal self-attention core, built by
    *composing* the four verified primitives: attention_scores -> causal_mask
    -> softmax_op -> attention_output. Q,K,V (B,H,T,d) -> O (B,H,T,d)."""
    S, bw_scores = attention_scores(Q, K)
    Sm, bw_mask = causal_mask(S, fill=-1e9)
    P, bw_softmax = softmax_op(Sm)
    O, bw_out = attention_output(P, V)
    def bwd(dO):
        g_out = bw_out(dO)                     # {'P', 'V'}
        g_sm = bw_softmax(g_out['P'])          # {'x': dSm}
        g_mask = bw_mask(g_sm['x'])            # {'S': dS}
        g_sc = bw_scores(g_mask['S'])          # {'Q', 'K'}
        return {'Q': g_sc['Q'], 'K': g_sc['K'], 'V': g_out['V']}
    return O, bwd


def mha(x, W_attn, b_attn, W_proj, b_proj, H):
    """Items 15 + 20 + 21. Full multi-head causal self-attention module.
    x (B,T,D) -> fused QKV projection -> head split -> sdpa_causal -> head
    merge -> output projection -> Y (B,T,D)."""
    B, T, D = x.shape
    d = D // H

    qkv = x @ W_attn + b_attn                  # (B,T,3D)
    Qf, Kf, Vf = np.split(qkv, 3, axis=-1)     # each (B,T,D)

    def split_heads(z):                        # (B,T,D) -> (B,H,T,d)
        return z.reshape(B, T, H, d).transpose(0, 2, 1, 3)

    Q, K, V = split_heads(Qf), split_heads(Kf), split_heads(Vf)
    O, bw_sdpa = sdpa_causal(Q, K, V)          # (B,H,T,d)
    O_merged = O.transpose(0, 2, 1, 3).reshape(B, T, D)
    Y = O_merged @ W_proj + b_proj             # (B,T,D)

    def bwd(dY):
        dO_merged = dY @ W_proj.T
        dW_proj = O_merged.reshape(-1, D).T @ dY.reshape(-1, D)
        db_proj = dY.reshape(-1, D).sum(axis=0)

        dO = dO_merged.reshape(B, T, H, d).transpose(0, 2, 1, 3)   # (B,H,T,d)
        g = bw_sdpa(dO)                                            # {'Q','K','V'}

        def merge_heads(t):                    # (B,H,T,d) -> (B,T,D)
            return t.transpose(0, 2, 1, 3).reshape(B, T, D)

        dqkv = np.concatenate([merge_heads(g['Q']),
                               merge_heads(g['K']),
                               merge_heads(g['V'])], axis=-1)       # (B,T,3D)
        dx = dqkv @ W_attn.T
        dW_attn = x.reshape(-1, D).T @ dqkv.reshape(-1, 3 * D)
        db_attn = dqkv.reshape(-1, 3 * D).sum(axis=0)
        return {'x': dx, 'W_attn': dW_attn, 'b_attn': db_attn,
                'W_proj': dW_proj, 'b_proj': db_proj}
    return Y, bwd


# --------------------------------------------------------------------------
# Tier 7: composition
# --------------------------------------------------------------------------

# The 12 learnable parameters of one pre-LN transformer block, in a fixed order.
BLOCK_PARAMS = ['ln1_g', 'ln1_b', 'W_attn', 'b_attn', 'W_proj', 'b_proj',
                'ln2_g', 'ln2_b', 'W1', 'mlp_b1', 'W2', 'mlp_b2']


def transformer_block(x, ln1_g, ln1_b, W_attn, b_attn, W_proj, b_proj,
                      ln2_g, ln2_b, W1, mlp_b1, W2, mlp_b2, H, eps=1e-5):
    """Item 22. Pre-LN transformer block, composed entirely from verified ops:
        a = LN1(x);  u = MHA(a);  r = x + u
        c = LN2(r);  m = Linear2(GeLU(Linear1(c)));  y = r + m
    The residual rule (y = x + f(x)  =>  dx = dy + f.backward(dy)) is exercised
    at both junctions: dr sums the residual-2 skip with the MLP branch, and dx
    sums the residual-1 skip with the attention branch."""
    a, bw_ln1 = layernorm(x, ln1_g, ln1_b, eps)
    u, bw_mha = mha(a, W_attn, b_attn, W_proj, b_proj, H)
    r = x + u
    c, bw_ln2 = layernorm(r, ln2_g, ln2_b, eps)
    h1, bw_lin1 = linear(c, W1, mlp_b1)
    h2, bw_gelu = gelu(h1)
    m, bw_lin2 = linear(h2, W2, mlp_b2)
    y = r + m

    def bwd(dy):
        # residual 2: y = r + m   -> dm = dy, and dy also skips around to r
        g_lin2 = bw_lin2(dy)                 # {'x': dh2, 'W': dW2, 'b': dmlp_b2}
        g_gelu = bw_gelu(g_lin2['x'])         # {'x': dh1}
        g_lin1 = bw_lin1(g_gelu['x'])         # {'x': dc, 'W': dW1, 'b': dmlp_b1}
        g_ln2 = bw_ln2(g_lin1['x'])           # {'x','gamma','beta'}
        dr = dy + g_ln2['x']                  # residual-2 skip + MLP branch

        # residual 1: r = x + u   -> du = dr, and dr also skips around to x
        g_mha = bw_mha(dr)                    # {'x': da, 'W_attn',...}
        g_ln1 = bw_ln1(g_mha['x'])            # {'x','gamma','beta'}
        dx = dr + g_ln1['x']                  # residual-1 skip + attention branch

        return {'x': dx,
                'ln1_g': g_ln1['gamma'], 'ln1_b': g_ln1['beta'],
                'W_attn': g_mha['W_attn'], 'b_attn': g_mha['b_attn'],
                'W_proj': g_mha['W_proj'], 'b_proj': g_mha['b_proj'],
                'ln2_g': g_ln2['gamma'], 'ln2_b': g_ln2['beta'],
                'W1': g_lin1['W'], 'mlp_b1': g_lin1['b'],
                'W2': g_lin2['W'], 'mlp_b2': g_lin2['b']}
    return y, bwd


def tiny_gpt(ids, targets, W_e, W_p, lnf_g, lnf_b, H, n_blocks, eps=1e-5, **block_kw):
    """Item 23. Full GPT-2 forward/backward, end to end:
        e_tok = W_e[ids];  e_pos = W_p[:T];  x = e_tok + e_pos
        x = block_0(x); ...; x = block_{N-1}(x)
        xf = LN_final(x);  logits = xf @ W_e^T  (TIED);  L = softmax_CE(logits)
    The whole point: W_e is touched twice in the backward - the lm_head writes
    it, the token-embedding scatter-add accumulates into it. Both halves are
    required, and the W_e gradient check fails if either is missing."""
    B, T = ids.shape
    D = W_e.shape[1]
    V = W_e.shape[0]

    e_tok = W_e[ids]                           # (B,T,D)
    e_pos = W_p[:T]                            # (T,D), broadcasts over batch
    x = e_tok + e_pos

    block_bwds = []
    for i in range(n_blocks):
        bp = {pn: block_kw[f'b{i}_{pn}'] for pn in BLOCK_PARAMS}
        x, bw = transformer_block(x, H=H, eps=eps, **bp)
        block_bwds.append(bw)

    xf, bw_lnf = layernorm(x, lnf_g, lnf_b, eps)
    logits = xf @ W_e.T                        # (B,T,V) -- weight tied
    L, bw_ce = softmax_ce_fused(logits, targets)

    def bwd(dL):
        g_ce = bw_ce(dL)
        dlogits = g_ce['logits']               # (B,T,V)

        # tied lm_head backward (item 14, Part A) -- INITIALISE dW_e here
        dxf = dlogits @ W_e
        dW_e = dlogits.reshape(-1, V).T @ xf.reshape(-1, D)        # weight-tie #1

        g_lnf = bw_lnf(dxf)
        dx = g_lnf['x']

        grads = {'lnf_g': g_lnf['gamma'], 'lnf_b': g_lnf['beta']}
        for i in reversed(range(n_blocks)):
            g_block = block_bwds[i](dx)
            dx = g_block['x']
            for pn in BLOCK_PARAMS:
                grads[f'b{i}_{pn}'] = g_block[pn]

        # embedding sum: x = e_tok + e_pos  -> gradient copies to both
        de_tok = dx
        de_pos = dx
        dW_p = np.zeros_like(W_p)
        dW_p[:T] = de_pos.sum(axis=0)

        # token-embedding scatter-add -- ACCUMULATE into the same dW_e buffer
        np.add.at(dW_e, ids, de_tok)                              # weight-tie #2

        grads['W_e'] = dW_e
        grads['W_p'] = dW_p
        return grads
    return L, bwd
