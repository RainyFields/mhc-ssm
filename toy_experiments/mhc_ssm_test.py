import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import math
import time
import os
import sys
import datetime
import argparse
import random
import json
import matplotlib.pyplot as plt

# ==========================================
# 0. Configuration
# ==========================================
# Base results directory
RESULT_DIR = "/work/lei/mhc-ssm/toy_experiments/results"
# Checkpoints specific directory
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

CONFIG = {
    "D_MODEL": 256,       
    "N_LAYERS": 4,        
    "VOCAB": 64,          
    "STREAMS": 4,         
    "BS": 32,
    "SEQ": 8,           
    "STEPS": 100,       
    "LR": 1e-3
}

# ==========================================
# 1. Mamba-2 Block
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

        # A: Initialized with ONES (Stability)
        self.A_log = nn.Parameter(torch.ones(self.n_heads) * -4.0) 
        # D: ONES (Pass-through signal)
        self.D = nn.Parameter(torch.ones(self.n_heads))
        # dt: Initialized with RAND (Uniform)
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
            
        y = torch.stack(ys, dim=1) # Shape: [B, L, H, P]
        
        # --- THE FIX: Apply D Skip Connection ---
        # "Classical" Mamba connection: y = SSM(x) + D*x
        # Reshape D to broadcast: [H] -> [1, 1, H, 1]
        y = y + (x * self.D.view(1, 1, self.n_heads, 1))
        # ----------------------------------------

        y = y.reshape(B_size, L, -1)
        y = y * F.silu(z) # Gating happens AFTER D
        return self.out_proj(y)

# ==========================================
# 2. Connection Mechanisms
# ==========================================
class SinkhornMixer(nn.Module):
    def __init__(self, n_streams, iters=5):
        super().__init__()
        self.iters = iters
        # RANDN: Break symmetry in mixing
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
        
        # RANDN: Break symmetry in routing weights
        self.W_pre = nn.Parameter(torch.randn(n_streams, 1) * 0.01)

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

    def forward(self, H, layer_fn):
        H_res = H # Identity
        x_flat = H.flatten(2, 3)
        x_out = layer_fn(x_flat)
        H_update = x_out.view(H.shape[0], H.shape[1], self.n_streams, self.d_sub)
        return H_res + H_update

# ==========================================
# 3. Models
# ==========================================
class Model_Standard(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([Mamba2SimpleBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = x + layer(x)
        return self.head(self.norm(x))

class Model_mHC(nn.Module):
    def __init__(self, d_model, n_layers, vocab_size, n_streams=4, use_mixer=True):
        super().__init__()
        self.n_streams = n_streams
        self.d_sub = d_model // n_streams
        self.embed = nn.Embedding(vocab_size, d_model)
        self.mamba_layers = nn.ModuleList([Mamba2SimpleBlock(d_model) for _ in range(n_layers)])
        
        if use_mixer:
            self.mhc_connects = nn.ModuleList([MHC_Connection(d_model, n_streams) for _ in range(n_layers)])
        else:
            self.mhc_connects = nn.ModuleList([NoMixer_Connection(d_model, n_streams) for _ in range(n_layers)])
            
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, L = x.shape
        x = self.embed(x)
        H = x.view(B, L, self.n_streams, self.d_sub)
        for layer, connect in zip(self.mamba_layers, self.mhc_connects):
            H = connect(H, layer)
        return self.head(self.norm(H.flatten(2, 3)))

# ==========================================
# 4. Harness & Parallel Worker
# ==========================================
def get_batch(bs, seq_len, vocab_size, device):
    data = torch.randint(0, vocab_size, (bs, seq_len))
    split = seq_len // 2
    data[:, split:] = data[:, :split]
    return data[:, :-1].to(device), data[:, 1:].to(device)

def train_worker(rank, model_type, gpu_id, seed, run_id, return_dict):
    """
    Worker function to run one model on one GPU.
    """
    # 1. Setup Device and Seed
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    print(f"[Worker {rank}] Starting {model_type} on GPU {gpu_id} with Seed {seed}")
    
    # 2. Initialize Model
    if model_type == "standard":
        model = Model_Standard(CONFIG["D_MODEL"], CONFIG["N_LAYERS"], CONFIG["VOCAB"]).to(device)
    elif model_type == "mhc":
        model = Model_mHC(CONFIG["D_MODEL"], CONFIG["N_LAYERS"], CONFIG["VOCAB"], CONFIG["STREAMS"], use_mixer=True).to(device)
    elif model_type == "mhc_nomixer":
        model = Model_mHC(CONFIG["D_MODEL"], CONFIG["N_LAYERS"], CONFIG["VOCAB"], CONFIG["STREAMS"], use_mixer=False).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    optim = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    loss_fn = nn.CrossEntropyLoss()
    history = {"step": [], "loss": [], "acc": []}
    
    # 3. Training Loop
    start_time = time.time()
    for i in range(CONFIG["STEPS"]):
        optim.zero_grad()
        x, y = get_batch(CONFIG["BS"], CONFIG["SEQ"], CONFIG["VOCAB"], device)
        logits = model(x)
        
        # --- COPY TASK MASKING ---
        split_idx = CONFIG["SEQ"] // 2
        valid_logits = logits[:, split_idx-1 : -1, :]
        valid_y = y[:, split_idx-1 : -1]
        
        loss = loss_fn(valid_logits.reshape(-1, CONFIG["VOCAB"]), valid_y.reshape(-1))
        preds = valid_logits.argmax(dim=-1)
        acc = (preds == valid_y).float().mean()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        
        if i % 100 == 0 or i == CONFIG["STEPS"] - 1:
            history["step"].append(i)
            history["loss"].append(loss.item())
            history["acc"].append(acc.item())
            if i % 1000 == 0:
                print(f"[Worker {rank} | {model_type}] Step {i}: Loss {loss.item():.4f} | Acc {acc.item():.4f}")

    total_time = time.time() - start_time
    print(f"[Worker {rank}] Finished in {total_time:.2f}s")
    
    # 4. Save Checkpoint
    ckpt_name = f"{run_id}_{model_type}_seed{seed}.pt"
    ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Worker {rank}] Checkpoint saved: {ckpt_path}")

    # 5. Return Results
    return_dict[rank] = {
        "model_type": model_type,
        "seed": seed,
        "history": history,
        "ckpt": ckpt_name
    }

def log_experiment(run_id, res1, res2): 
    # Unpack results
    name1, h1, s1 = res1["model_type"], res1["history"], res1["seed"]
    name2, h2, s2 = res2["model_type"], res2["history"], res2["seed"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # --- PLOT 1: LOSS ---
    ax1.plot(h1['step'], h1['loss'], label=f'{name1} (s:{s1})', color='blue')
    ax1.plot(h2['step'], h2['loss'], label=f'{name2} (s:{s2})', color='orange')
    ax1.set_title(f"Loss (Run {run_id})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()

    # --- PLOT 2: ACCURACY ---
    ax2.plot(h1['step'], h1['acc'], label=f'{name1} (s:{s1})', color='blue')
    ax2.plot(h2['step'], h2['acc'], label=f'{name2} (s:{s2})', color='orange')
    ax2.set_title(f"Accuracy (Seq {CONFIG['SEQ']})")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{run_id}_plot.png"))
    plt.close()

    # --- LOGGING TO MARKDOWN ---
    md_path = os.path.join(RESULT_DIR, "experiments_log.md")
    
    # Serialize config to JSON string for the log
    config_json = json.dumps(CONFIG)

    with open(md_path, "a") as f:
        if os.stat(md_path).st_size == 0:
            f.write("| Run ID | Model 1 | Seed 1 | Acc 1 | Model 2 | Seed 2 | Acc 2 | Config |\n")
            f.write("|---|---|---|---|---|---|---|---|\n")
        
        acc1 = h1['acc'][-1] if h1['acc'] else 0.0
        acc2 = h2['acc'][-1] if h2['acc'] else 0.0
        
        f.write(f"| {run_id} | {name1} | {s1} | {acc1:.4f} | {name2} | {s2} | {acc2:.4f} | {config_json} |\n")

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model1", type=str, default="mhc", choices=["mhc", "mhc_nomixer", "standard"])
    parser.add_argument("--model2", type=str, default="standard", choices=["mhc", "mhc_nomixer", "standard"])
    parser.add_argument("--gpu1", type=int, default=0)
    parser.add_argument("--gpu2", type=int, default=1)
    args = parser.parse_args()

    # Create Run ID
    run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    
    # Generate Random Seeds
    seed1 = random.randint(1000, 9999)
    seed2 = random.randint(1000, 9999)

    print(f"--- Experiment: {run_id} ---")
    print(f"Config: {CONFIG}")
    print(f"Job 1: {args.model1} on GPU {args.gpu1} (Seed {seed1})")
    print(f"Job 2: {args.model2} on GPU {args.gpu2} (Seed {seed2})")

    # Use Multiprocessing Manager
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    # Create Processes
    p1 = mp.Process(target=train_worker, args=(0, args.model1, args.gpu1, seed1, run_id, return_dict))
    p2 = mp.Process(target=train_worker, args=(1, args.model2, args.gpu2, seed2, run_id, return_dict))

    # Start and Join
    p1.start()
    p2.start()
    p1.join()
    p2.join()

    # Logging
    print("\nTraining complete. Logging results...")
    if 0 in return_dict and 1 in return_dict:
        log_experiment(run_id, return_dict[0], return_dict[1])
        print(f"Results logged to {RESULT_DIR}/experiments_log.md")
        print(f"Checkpoints saved to {CHECKPOINT_DIR}")
    else:
        print("Error: Workers did not return data successfully.")