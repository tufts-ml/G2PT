import os
import time
import math
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from datasets_utils import get_datasets
from transformers import AutoTokenizer
# from tokenization_g2pt_fast import G2PTTokenizerFast
from torch.utils.data.distributed import DistributedSampler
from model import GPTConfig, GPT
from transformers import default_data_collator

torch._dynamo.config.optimize_ddp = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -----------------------------------------------------------------------------
# Default configuration values for training a GPT-2 model on OpenWebText

# I/O settings
out_dir = 'out'  # Directory to save outputs
eval_interval = 1000  # Interval for evaluation
log_interval = 10  # Interval for logging
eval_iters = 200  # Number of iterations for evaluation
always_save_checkpoint = False  # Save checkpoint after each eval if True
init_from = 'scratch'  # Options: 'scratch', 'resume', 'gpt2*'

# Weights & Biases (wandb) logging settings
wandb_log = False  # Enable wandb logging if True
wandb_project = 'g2pt'  # Wandb project name
wandb_run_name = None  # Wandb run name

# Data settings
dataset = None  # Dataset to be used
gradient_accumulation_steps = 5 * 8  # Simulate larger batch sizes
batch_size = 12  # Micro-batch size if gradient_accumulation_steps > 1
block_size = 1024  # Block size for model input
vocab_size = None  # Vocabulary size
ordering = 'bfs'

# Model architecture settings
n_layer = 12  # Number of layers
n_head = 12  # Number of attention heads
n_embd = 768  # Embedding size
dropout = 0.0  # Dropout rate; 0 for pretraining, 0.1+ for finetuning
bias = False  # Use bias in LayerNorm and Linear layers if True
model_name = 'base'

# AdamW optimizer settings
learning_rate = 1e-4  # Maximum learning rate
max_iters = 300000  # Total number of training iterations
weight_decay = 1e-1  # Weight decay for optimizer
beta1 = 0.9  # Beta1 for AdamW
beta2 = 0.95  # Beta2 for AdamW
grad_clip = 1.0  # Gradient clipping value; disable if 0.0

# Learning rate decay settings
decay_lr = True  # Enable learning rate decay if True
warmup_iters = 2000  # Number of warmup iterations
lr_decay_iters = 300000  # Iterations for learning rate decay
min_lr = 1e-5  # Minimum learning rate

# Distributed Data Parallel (DDP) settings
backend = 'nccl'  # Backend for DDP; options: 'nccl', 'gloo', etc.

# System settings
device = 'cuda'  # Device for training; options: 'cpu', 'cuda', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # Data type
compile = False  # Compile model with PyTorch 2.0 for speed if True

# -----------------------------------------------------------------------------
# Load additional configuration from external file
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # Override settings from command line or config file
config = {k: globals()[k] for k in config_keys}  # Configuration dictionary for logging
# -----------------------------------------------------------------------------
if wandb_log:
    wandb_run_name = f"{dataset}-{model_name}-{ordering}"
out_dir = f'results/{wandb_run_name}'
# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter},{gradient_accumulation_steps},{ddp_world_size}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# data preparation
tokenizer = AutoTokenizer.from_pretrained(f'tokenizers/{dataset}')
train_dataset, eval_dataset = get_datasets(dataset, tokenizer, ordering)

def data_collate_fn(features):
    # datasets output datapoint with max length, we need to truncate to the max length of the batch (by checking the attention mask)
    features = default_data_collator(features)
    seq_len = features['attention_mask'].sum(-1)
    max_len = seq_len.max()
    features = {k:v[...,:max_len] for k,v in features.items()}
    return features

train_sampler = DistributedSampler(train_dataset,shuffle=True) if ddp else None
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    pin_memory=True,
    drop_last=False,
    num_workers=8,
    collate_fn=data_collate_fn
)

eval_sampler = DistributedSampler(eval_dataset) if ddp else None 
eval_loader = torch.utils.data.DataLoader(
    eval_dataset,
    batch_size=batch_size,
    sampler=eval_sampler,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    num_workers=8,
    collate_fn=data_collate_fn
)


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    model_args['vocab_size'] = vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
    
# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', eval_loader)]:
        losses = torch.zeros(eval_iters)
        num_eval_iters = 0
        while True:
            for data in loader:
                X, Y, Y_mask = data['input_ids'][:,:-1], data['labels'][:,1:], data['attention_mask'][:,1:]
                X = X.to(device)
                Y = Y.to(device)
                Y_mask = Y_mask.to(device)
                with ctx:
                    logits, loss = model(X, Y, Y_mask)
                losses[num_eval_iters] = loss.item()
                num_eval_iters += 1
                if num_eval_iters >= eval_iters:
                    break
            
            if num_eval_iters >= eval_iters:
                break
            
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop

t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0

micro_step = 0
while True:
    for data in train_loader:
        X, Y, Y_mask = data['input_ids'][:,:-1], data['labels'][:,1:], data['attention_mask'][:,1:]
        X = X.to(device)
        Y = Y.to(device)
        Y_mask = Y_mask.to(device)

        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        
        with ctx:
            logits, loss = model(X, Y, Y_mask)
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
        micro_step += 1

        if micro_step == gradient_accumulation_steps:
            micro_step = 0
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            if iter_num % log_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                lossf = loss.item() * gradient_accumulation_steps
                if local_iter_num >= 5: # let the training loop settle a bit
                    mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            iter_num += 1
            local_iter_num += 1

            # termination conditions
            if iter_num > max_iters:
                break
            # determine and set the learning rate for this iteration
            lr = get_lr(iter_num) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % eval_interval == 0 and master_process and iter_num != 0:
                losses = estimate_loss()
                print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": lr,
                        "mfu": running_mfu*100, # convert to percentage
                    })
                if losses['val'] < best_val_loss or always_save_checkpoint:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config,
                        }
                        print(f"saving checkpoint to {out_dir}")
                        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()

