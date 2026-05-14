"""
Tiny CPU baseline: 100 steps of nanoGPT on a small OpenWebText subset.

Mirrors the README's CPU-friendly recipe (4L/4H/128d, block=64, batch=12)
but trained against the OWT pipeline (data/openwebtext/*.bin produced by
prepare_subset.py) so the loss curve is comparable to a small OWT baseline.
"""

out_dir = 'out-owt-baseline'
eval_interval = 100        # eval only at start and at the end of the run
eval_iters = 20            # quick noisy estimate
log_interval = 1           # log every step so we get a full loss curve
always_save_checkpoint = False

wandb_log = False

dataset = 'openwebtext'
gradient_accumulation_steps = 1   # one micro-batch per step on CPU
batch_size = 12
block_size = 64

# tiny model
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0
bias = False

# 100-step run with short warmup
learning_rate = 1e-3
max_iters = 100
lr_decay_iters = 100
min_lr = 1e-4
warmup_iters = 10
beta2 = 0.99               # README recommends 0.99 for very small models

weight_decay = 1e-1
beta1 = 0.9
grad_clip = 1.0
decay_lr = True

# CPU
device = 'cpu'
compile = False
dtype = 'float32'
