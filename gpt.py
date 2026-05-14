"""
gpt.py - The full GPT-2 model, assembled from layers.py on the autograd Tensor.

Mirrors nanoGPT's model.py `GPT`:

    token embedding (wte) + positional embedding (wpe)
      -> N TransformerBlocks
      -> final LayerNorm (ln_f)
      -> lm_head  (WEIGHT-TIED to wte: one matrix, two uses)
      -> cross-entropy loss  (mean over B*T, == torch F.cross_entropy reduction='mean')

**Weight tying is explicit.** There is no separate `lm_head` parameter. The
forward calls `tensor.lm_head(x, self.wte.weight)` - the *same* Tensor the token
embedding looks up. The autograd graph therefore accumulates *both* gradient
contributions (embedding scatter-add + output-projection matmul) into that one
Tensor automatically. `load_state_dict` skips the `lm_head.weight` key after
checking it equals `transformer.wte.weight`.

**GeLU.** nanoGPT's `model.py` uses `nn.GELU()` - the EXACT erf GeLU - so this
model uses the exact erf form (see `derivation.md` item 4b, `tests/ops.py:gelu`).
The Step-4 brief mentions "tanh-approx GeLU"; that is the *older* GPT-2 / nanoGPT
form, and it would miss this parity gate by ~1e-3. The exact form is what the
checkpoint in `fixtures/parity_batch.pt` was trained with. See
`tests/model_parity.md` and `nanogpt/roadblocks.md` (2026-05-14 entry).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tensor as T                                                    # noqa: E402
from layers import Module, Embedding, LayerNorm, TransformerBlock, _prefixed  # noqa: E402


class GPTConfig:
    """nanoGPT GPTConfig, as a plain class (kwargs match the fixture's model_args)."""

    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd,
                 bias=True, dropout=0.0):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.bias = bias
        # `dropout` is stored for fidelity but unused: we run dropout-free, which
        # equals nanoGPT at dropout=0.0 - the configuration of the parity fixture.
        self.dropout = dropout


class GPT(Module):
    def __init__(self, config):
        self.config = config
        self.wte = Embedding(config.vocab_size, config.n_embd)         # token embedding
        self.wpe = Embedding(config.block_size, config.n_embd)         # positional embedding
        self.h = [TransformerBlock(config.n_embd, config.n_head, bias=config.bias)
                  for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        # lm_head is weight-tied to wte - NOT a separate parameter (see module docstring).

    def __call__(self, idx, targets=None):
        """idx: int array (B, T). Returns (logits, loss); loss is None if targets is None."""
        B, Tn = idx.shape
        assert Tn <= self.config.block_size, \
            f"sequence length {Tn} exceeds block_size {self.config.block_size}"

        tok_emb = self.wte(idx)                       # (B, T, n_embd)
        pos_emb = self.wpe(np.arange(Tn))             # (T, n_embd), broadcasts over batch
        x = tok_emb + pos_emb
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = T.lm_head(x, self.wte.weight)        # weight tying: wte.weight reused

        if targets is None:
            return logits, None
        loss = T.softmax_cross_entropy(logits, targets)
        return logits, loss

    def parameters(self):
        p = {}
        p.update(_prefixed('wte', self.wte.parameters()))
        p.update(_prefixed('wpe', self.wpe.parameters()))
        for i, block in enumerate(self.h):
            p.update(_prefixed(f'h.{i}', block.parameters()))
        p.update(_prefixed('ln_f', self.ln_f.parameters()))
        return p

    def load_state_dict(self, sd):
        """Load a nanoGPT PyTorch state_dict ({key: array-like}) into this model.

        Handles: the `transformer.` prefix, LayerNorm `weight`/`bias` ->
        `gamma`/`beta`, Linear `weight` transpose ((out,in) -> our (in,out)),
        and `lm_head.weight` (skipped - weight-tied - after checking it equals
        `transformer.wte.weight`)."""
        my_params = self.parameters()
        loaded = set()
        for key, arr in sd.items():
            my_name, transpose = _torch_key_to_my(key)
            if my_name is None:
                continue                              # lm_head.weight / buffers
            arr = np.asarray(arr, dtype=np.float64)
            if transpose:
                arr = arr.T
            if my_name not in my_params:
                raise KeyError(f"state_dict key {key!r} -> {my_name!r} is not a model parameter")
            want = my_params[my_name].data.shape
            if arr.shape != want:
                raise ValueError(f"{my_name}: state_dict shape {arr.shape} != model shape {want}")
            my_params[my_name].data = np.ascontiguousarray(arr)
            loaded.add(my_name)

        missing = set(my_params) - loaded
        if missing:
            raise KeyError(f"parameters not present in state_dict: {sorted(missing)}")

        # explicit weight-tying check: lm_head.weight must equal transformer.wte.weight
        if 'lm_head.weight' in sd:
            lm = np.asarray(sd['lm_head.weight'], dtype=np.float64)
            if not np.array_equal(lm, self.wte.weight.data):
                raise ValueError("lm_head.weight != transformer.wte.weight - the state_dict "
                                 "violates the weight-tying assumption")
        return self


def _torch_key_to_my(key):
    """Map a nanoGPT state_dict / named_parameter key to (my_param_name, transpose).

    Returns (None, False) for keys we skip: `lm_head.weight` (weight-tied to
    `wte.weight`) and `*.attn.bias` (the causal-mask buffer, not a parameter)."""
    if key.startswith('lm_head'):
        return None, False                            # weight-tied to wte.weight
    if key.endswith('.attn.bias') or key.endswith('.attn.masked_bias'):
        return None, False                            # causal-mask buffer, not a parameter
    k = key[len('transformer.'):] if key.startswith('transformer.') else key
    parts = k.split('.')
    last, module = parts[-1], parts[-2]
    if module in ('wte', 'wpe'):                      # Embedding: weight stays weight
        return k, False
    if module in ('ln_1', 'ln_2', 'ln_f'):            # LayerNorm: weight/bias -> gamma/beta
        my_last = 'gamma' if last == 'weight' else 'beta'
        return '.'.join(parts[:-1] + [my_last]), False
    if module in ('c_attn', 'c_proj', 'c_fc'):        # Linear: weight -> W (transposed), bias -> b
        my_last = 'W' if last == 'weight' else 'b'
        return '.'.join(parts[:-1] + [my_last]), (last == 'weight')
    raise ValueError(f"unrecognized nanoGPT state_dict key: {key!r}")
