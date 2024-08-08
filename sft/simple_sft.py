import os
import time
import math
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataset_sft import SFTDataset

# Set visible GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def jsonl_to_dataframe(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def get_lr(it, warmup_iters, lr_decay_iters, learning_rate, min_lr):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def train_epoch(epoch, train_loader, model, optimizer, scaler, device, iter_per_epoch, gradient_accumulation_steps, log_interval, logger, learning_rate, decay_lr, warmup_iters, lr_decay_iters, min_lr, grad_clip, ctx, ddp):
    model.train()
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(device)
        Y = Y.to(device)
        loss_mask = loss_mask.to(device)
        lr = get_lr(epoch * iter_per_epoch + step, warmup_iters, lr_decay_iters, learning_rate, min_lr) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if ddp:
            model.require_backward_grad_sync = (gradient_accumulation_steps - 1 == 0)
        with ctx:
            logits = model(X, labels=Y).logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-100, reduction='none')
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()
        scaler.scale(loss).backward()
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if step % log_interval == 0:
            elapsed_time = time.time() - start_time
            logger.info(f'Epoch: [{epoch}/{max_epoch}] Step: [{step}/{iter_per_epoch}] Loss: {loss.item():.4f} LR: {optimizer.param_groups[0]["lr"]:.7f} Elapsed Time: {elapsed_time:.2f}s')

@torch.no_grad()
def valid_epoch(model, val_loader, device, ctx):
    model.eval()
    losses = []
    for step, (X, Y, loss_mask) in enumerate(val_loader):
        X = X.to(device)
        Y = Y.to(device)
        loss_mask = loss_mask.to(device)
        with ctx:
            outputs = model(X, labels=Y)
            logits = outputs.logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-100, reduction='none')
        losses.append(loss.mean().item())
    model.train()
    return np.mean(losses)

if __name__ == "__main__":
    out_dir = 'out'
    max_epoch = 10
    log_interval = 50
    eval_interval = 1
    eval_iters = 200
    eval_only = False
    always_save_checkpoint = True
    gradient_accumulation_steps = 1
    batch_size = 8
    max_seq_len = 512
    learning_rate = 2e-5
    weight_decay = 1e-4
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    decay_lr = True
    warmup_iters = 1000
    lr_decay_iters = 50000
    min_lr = 1e-6
    backend = 'nccl'
    device = 'cuda'
    dtype = 'float16'
    compile = False
    save_dir = os.path.join(out_dir, 'sft')
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(os.path.join(save_dir, 'log.log'))

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        master_process = True

    tokens_per_iter = gradient_accumulation_steps * batch_size * max_seq_len
    if master_process:
        print(f"tokens per iteration: {tokens_per_iter:,}")

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    best_val_loss = float('inf')

    df1 = jsonl_to_dataframe("dataset/simple_itern_output.jsonl")
    df2=  jsonl_to_dataframe("dataset/train.jsonl")
    df= pd.concat([df1,df2],ignore_index=True)
    df = df.replace("####", "##answer:", regex=True)
    df = df.sample(frac=1.0).reset_index(drop=True)
    tokenizer = AutoTokenizer.from_pretrained("models/internlm2-chat-1_8b", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("models/internlm2-chat-1_8b", trust_remote_code=True)
    train_ds = SFTDataset(df, tokenizer, max_length=512)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2))

    model.to(device)

    if compile:
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    raw_model = model.module if ddp else model

    for epoch in range(max_epoch):
        train_epoch(epoch, train_loader, model, optimizer, scaler, device, iter_per_epoch, gradient_accumulation_steps, log_interval, logger, learning_rate, decay_lr, warmup_iters, lr_decay_iters, min_lr, grad_clip, ctx, ddp)
        if (epoch + 1) % eval_interval == 0:
            val_loss = valid_epoch(model, train_loader, device, ctx)
            logger.info(f"Validation Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"New best validation loss: {best_val_loss:.4f}")
                torch.save(raw_model.state_dict(), os.path.join(save_dir, 'best.pth'))

        # if always_save_checkpoint or (epoch + 1) % eval_interval == 0:
        #     torch.save(raw_model.state_dict(), os.path.join(save_dir, f'epoch_{epoch+1}.pth'))

    if ddp:
        destroy_process_group()
