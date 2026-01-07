import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import csv
import datetime
import time
import json
import matplotlib.pyplot as plt
import wandb
import numpy as np

# ==========================================
# 1. Modular Components (Model Definitions)
# ==========================================

class Mamba2SimpleBlock(nn.Module):
    def __init__(self, d_model, d_state=64, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.n_heads = 4 
        self.d_proj_out = (2 * self.d_inner) + (2 * self.n_heads * d_state) + self.n_heads
        self.in_proj = nn.Linear(d_model, self.d_proj_out)

        self.A_log = nn.Parameter(torch.ones(self.n_heads) * -4.0) 
        self.D = nn.Parameter(torch.ones(self.n_heads))
        self.dt_bias = nn.Parameter(torch.rand(self.n_heads) - 4.0)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, u):
        u_norm = self.norm(u)
        B_size, L, _ = u_norm.shape
        zxbcdt = self.in_proj(u_norm)
        d_in = self.d_inner
        n_heads = self.n_heads
        d_state = self.d_state
        
        z = zxbcdt[:, :, :d_in]
        x = zxbcdt[:, :, d_in : 2*d_in]
        dt = zxbcdt[:, :, 2*d_in : 2*d_in + n_heads]
        BC = zxbcdt[:, :, 2*d_in + n_heads :]
        
        BC = BC.view(B_size, L, n_heads, 2, d_state)
        B_mat, C_mat = BC[:, :, :, 0], BC[:, :, :, 1]
        x = x.view(B_size, L, n_heads, -1)
        
        dt = F.softplus(dt + self.dt_bias)
        A = -torch.exp(self.A_log)
        dtA = torch.exp(dt * A.view(1, 1, -1))
        dtB = torch.einsum('blh, blhn -> blhn', dt, B_mat)
        
        h = torch.zeros(B_size, n_heads, d_state, x.shape[-1], device=u.device)
        ys = []
        
        # SSM Scan
        for t in range(L):
            dtA_t = dtA[:, t, :, None, None]
            dtB_t = dtB[:, t, :, :, None]
            x_t = x[:, t, :, None, :]
            C_t = C_mat[:, t, :, None, :]

            h = h * dtA_t + torch.matmul(dtB_t, x_t)
            y_t = torch.matmul(C_t, h).squeeze(-2)
            ys.append(y_t)
            
        y = torch.stack(ys, dim=1).reshape(B_size, L, -1)
        y = y * F.silu(z)
        return self.out_proj(y)
 
class SinkhornMixer(nn.Module):
    def __init__(self, n_streams, iters=5):
        super().__init__()
        self.iters = iters
        self.logits = nn.Parameter(torch.randn(n_streams, n_streams) * 0.1)
        self.eye = torch.eye(n_streams)

    def forward(self):
        M = F.sigmoid(self.logits) + self.eye.to(self.logits.device)
        for _ in range(self.iters):
            M = M / (M.sum(dim=1, keepdim=True) + 1e-6)
            M = M / (M.sum(dim=0, keepdim=True) + 1e-6)
        return M

class MHC_Connection(nn.Module):
    def __init__(self, d_model, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.d_sub = d_model // n_streams
        self.mixer = SinkhornMixer(n_streams)
        self.W_pre = nn.Parameter(torch.ones(n_streams, 1) / n_streams)

    def forward(self, H, layer_fn):
        M = self.mixer()
        H_res = torch.einsum('bsid, ji -> bsjd', H, M)
        x_flat = H.flatten(2, 3) 
        x_out = layer_fn(x_flat)
        H_update = x_out.view(H.shape[0], H.shape[1], self.n_streams, self.d_sub)
        return H_res + H_update

class NoMixer_Connection(nn.Module):
    def __init__(self, d_model, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.d_sub = d_model // n_streams
        # No Sinkhorn Mixer

    def forward(self, H, layer_fn):
        # Identity Path: NO MIXING (M = Identity)
        H_res = H 
        # Layer Path
        x_flat = H.flatten(2, 3) 
        x_out = layer_fn(x_flat)
        H_update = x_out.view(H.shape[0], H.shape[1], self.n_streams, self.d_sub)
        return H_res + H_update

class GenericModel(nn.Module):
    def __init__(self, args, model_type):
        super().__init__()
        self.args = args
        self.model_type = model_type
        self.embed = nn.Embedding(args.vocab, args.d_model)
        self.norm = nn.RMSNorm(args.d_model)
        self.head = nn.Linear(args.d_model, args.vocab)
        
        self.layers = nn.ModuleList([Mamba2SimpleBlock(args.d_model) for _ in range(args.n_layers)])
        
        self.connections = None
        if "mhc" in model_type:
            self.n_streams = args.streams
            self.d_sub = args.d_model // args.streams
            if model_type == "mhc":
                self.connections = nn.ModuleList([MHC_Connection(args.d_model, args.streams) for _ in range(args.n_layers)])
            elif model_type == "nomixer":
                self.connections = nn.ModuleList([NoMixer_Connection(args.d_model, args.streams) for _ in range(args.n_layers)])

    def forward(self, x):
        B, L = x.shape
        x = self.embed(x)
        
        if self.connections: 
            H = x.view(B, L, self.n_streams, self.d_sub)
            for layer, connect in zip(self.layers, self.connections):
                H = connect(H, layer)
            out = H.flatten(2, 3)
        else: 
            for layer in self.layers:
                x = x + layer(x)
            out = x
            
        return self.head(self.norm(out))

# ==========================================
# 2. Data Generators
# ==========================================

def get_batch_copy(bs, seq_len, vocab_size, device):
    data = torch.randint(0, vocab_size, (bs, seq_len))
    split = seq_len // 2
    data[:, split:] = data[:, :split] 
    return data[:, :-1].to(device), data[:, 1:].to(device)

def get_batch_mqar(bs, seq_len, vocab_size, device, num_pairs=8):
    if num_pairs * 2 + 2 > seq_len: num_pairs = seq_len // 4
    keys = torch.randint(0, vocab_size // 2, (bs, num_pairs))
    vals = torch.randint(vocab_size // 2, vocab_size, (bs, num_pairs))
    
    context = torch.stack([keys, vals], dim=2).view(bs, -1)
    
    query_idx = torch.randint(0, num_pairs, (bs, 1))
    query_key = keys.gather(1, query_idx)
    target_val = vals.gather(1, query_idx).squeeze(1)
    
    curr_len = context.shape[1]
    pad_len = seq_len - curr_len - 1
    padding = torch.zeros(bs, pad_len, dtype=torch.long)
    
    inputs = torch.cat([context, padding, query_key], dim=1).to(device)
    targets = target_val.to(device)
    return inputs, targets

def get_batch_fuzzy(bs, seq_len, vocab_size, device, needle_len=2):
    data = torch.randint(0, vocab_size, (bs, seq_len))
    needle = torch.randint(0, vocab_size, (bs, needle_len))
    pos = torch.randint(0, (seq_len // 2) - needle_len, (bs,))
    for i in range(bs):
        data[i, pos[i]:pos[i]+needle_len] = needle[i]
        
    data[:, -1] = needle[:, 0]
    inputs = data.to(device)
    targets = needle[:, 1].to(device) 
    return inputs, targets

# ==========================================
# 3. Training & Logging
# ==========================================

def train_model(args, model_type, run_name):
    # Initialize WandB
    wandb.init(project="mhc-final-experiments", name=run_name, config=vars(args), reinit=True)
    
    print(f"--> Starting: {model_type} | {run_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GenericModel(args, model_type).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    wandb.log({"n_params": n_params})
    print(f"    Params: {n_params/1e6:.4f}M")
    
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    
    log_file = os.path.join(args.result_dir, f"{run_name}.csv")
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'loss', 'acc', 'tps']) 

    history = {'step': [], 'loss': [], 'acc': [], 'tps': []}
    
    for step in range(args.steps):
        iter_start = time.time()
        optim.zero_grad()
        
        if args.task == 'copy':
            x, y = get_batch_copy(args.bs, args.seq, args.vocab, device)
            logits = model(x)
            split_idx = args.seq // 2
            valid_logits = logits[:, split_idx-1 : -1, :]
            valid_y = y[:, split_idx-1 : -1]
            loss = loss_fn(valid_logits.reshape(-1, args.vocab), valid_y.reshape(-1))
            preds = valid_logits.argmax(dim=-1)
            acc = (preds == valid_y).float().mean()
            
        elif args.task == 'mqar':
            x, y = get_batch_mqar(args.bs, args.seq, args.vocab, device)
            logits = model(x)
            last_token_logits = logits[:, -1, :]
            loss = loss_fn(last_token_logits, y)
            preds = last_token_logits.argmax(dim=-1)
            acc = (preds == y).float().mean()

        elif args.task == 'fuzzy_recall':
            x, y = get_batch_fuzzy(args.bs, args.seq, args.vocab, device)
            logits = model(x)
            last_token_logits = logits[:, -1, :]
            loss = loss_fn(last_token_logits, y)
            preds = last_token_logits.argmax(dim=-1)
            acc = (preds == y).float().mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        
        iter_end = time.time()
        tps = (args.bs * args.seq) / (iter_end - iter_start + 1e-6)
        
        if step % 50 == 0 or step == args.steps - 1:
            history['step'].append(step)
            history['loss'].append(loss.item())
            history['acc'].append(acc.item())
            history['tps'].append(tps)
            
            wandb.log({"loss": loss.item(), "accuracy": acc.item(), "tps": tps, "step": step})
            
            with open(log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, loss.item(), acc.item(), tps])
            print(f"\r{model_type} Step {step}: Loss {loss.item():.4f} | Acc {acc.item():.4f}", end="")

        # --- EARLY STOPPING CHECK ---
        if acc.item() >= args.target_acc:
            print(f"\n[Early Stop] Reached target accuracy {args.target_acc} at step {step}")
            # Log final state
            history['step'].append(step)
            history['loss'].append(loss.item())
            history['acc'].append(acc.item())
            history['tps'].append(tps)
            break

    # --- SAVE CHECKPOINT ---
    ckpt_name = f"{run_name}.pth"
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved model to {ckpt_path}")
    wandb.finish()
    
    return history

def plot_comparison(args, history_dict, title_suffix):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'standard': 'red', 'mhc': 'blue', 'nomixer': 'orange'}
    
    final_accs, avg_tps = {}, {}

    for name, hist in history_dict.items():
        c = colors.get(name, 'black')
        ax1.plot(hist['step'], hist['loss'], label=name, color=c)
        ax2.plot(hist['step'], hist['acc'], label=name, color=c)
        ax3.plot(hist['step'], hist['tps'], label=name, color=c, alpha=0.5)
        
        final_accs[name] = hist['acc'][-1] if hist['acc'] else 0.0
        avg_tps[name] = np.mean(hist['tps']) if hist['tps'] else 0.0

    ax1.set_title("Loss (Lower Better)"); ax2.set_title("Acc (Higher Better)"); ax3.set_title("Throughput (TPS)")
    for ax in [ax1, ax2, ax3]: ax.legend()
    
    plt.suptitle(f"Task: {args.task} | {title_suffix}")
    plt.savefig(os.path.join(args.result_dir, f"{args.exp_id}_plot.png"))

    md_path = os.path.join(args.result_dir, "experiments_log.md")
    config_summary = f"L={args.n_layers}, D={args.d_model}, SEQ={args.seq}, TASK={args.task}"
    std_tps = f"{int(avg_tps.get('standard', 0))}"
    mhc_tps = f"{int(avg_tps.get('mhc', 0))}"
    std_a = f"{final_accs.get('standard', 0):.4f}"
    mhc_a = f"{final_accs.get('mhc', 0):.4f}"
    
    with open(md_path, "a") as f:
        if os.stat(md_path).st_size == 0:
            f.write("| ID | Task | Std Acc | mHC Acc | Std TPS | mHC TPS | Config |\n")
            f.write("|---|---|---|---|---|---|---|\n")
        f.write(f"| {args.exp_id} | {args.task} | {std_a} | {mhc_a} | {std_tps} | {mhc_tps} | {config_summary} |\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=['std_vs_mhc', 'mhc_vs_nomixer'], default='std_vs_mhc')
    parser.add_argument("--task", type=str, choices=['copy', 'mqar', 'fuzzy_recall'], default='copy')
    
    # Directories
    parser.add_argument("--result_dir", type=str, default="/work/lei/mhc-ssm/toy_experiments/results")
    parser.add_argument("--ckpt_dir", type=str, default="/work/lei/mhc-ssm/toy_experiments/checkpoints")
    
    # Training Params
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--vocab", type=int, default=64)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10000) # Increased default
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--target_acc", type=float, default=0.9, help="Stop if acc reaches this")

    args = parser.parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    
    with open(os.path.join(args.result_dir, f"{args.exp_id}_config.json"), 'w') as f:
        json.dump(vars(args), f, indent=4)
        
    histories = {}
    if args.mode == 'std_vs_mhc':
        histories['standard'] = train_model(args, 'standard', f"{args.exp_id}_std")
        histories['mhc'] = train_model(args, 'mhc', f"{args.exp_id}_mhc")
    elif args.mode == 'mhc_vs_nomixer':
        histories['mhc'] = train_model(args, 'mhc', f"{args.exp_id}_mhc")
        histories['nomixer'] = train_model(args, 'nomixer', f"{args.exp_id}_nomixer")

    suffix = f"L={args.n_layers}, D={args.d_model}, SEQ={args.seq}"
    plot_comparison(args, histories, suffix)