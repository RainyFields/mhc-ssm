import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import sys
import datetime
import matplotlib.pyplot as plt

# ==========================================
# 0. Configuration (ABLATON SETTINGS)
# ==========================================
RESULT_DIR = "/work/lei/mhc-ssm/toy_experiments/results"
os.makedirs(RESULT_DIR, exist_ok=True)

# Using the "Stress Test" settings where mHC won (SEQ=512)
CONFIG = {
    "D_MODEL": 256,  # Small model
    "N_LAYERS": 4,
    "VOCAB": 64,
    "STREAMS": 4,
    "BS": 32,
    "SEQ": 256,      # HARD MODE
    "STEPS": 8000,   # Long training
    "LR": 1e-3
}

# ==========================================
# 1. Mamba Block (Standard)
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
        
        # Slicing the projection
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

# ==========================================
# 2. Connections (Real vs Ablation)
# ==========================================
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

        ###### to check: might be the difference, *0.01 instead of divided by n_streams
        self.W_pre = nn.Parameter(torch.ones(n_streams, 1) / n_streams)

    def forward(self, H, layer_fn):
        M = self.mixer()
        H_res = torch.einsum('bsid, ji -> bsjd', H, M) # SWAP STREAMS
        x_flat = H.flatten(2, 3)
        x_out = layer_fn(x_flat)
        H_update = x_out.view(H.shape[0], H.shape[1], self.n_streams, self.d_sub)
        return H_res + H_update

# --- ABLATION CLASS: No Mixer ---
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
# 4. Harness
# ==========================================
def get_batch(bs, seq_len, vocab_size):
    data = torch.randint(0, vocab_size, (bs, seq_len))
    split = seq_len // 2
    data[:, split:] = data[:, :split]
    return data[:, :-1].cuda(), data[:, 1:].cuda()

def run_experiment(model_type, run_id):
    print(f"\n--- Training {model_type.upper()} ---")
    
    if model_type == "standard":
        model = Model_Standard(CONFIG["D_MODEL"], CONFIG["N_LAYERS"], CONFIG["VOCAB"]).cuda()
    elif model_type == "mhc":
        model = Model_mHC(CONFIG["D_MODEL"], CONFIG["N_LAYERS"], CONFIG["VOCAB"], CONFIG["STREAMS"], use_mixer=True).cuda()
    elif model_type == "mhc_nomixer":
        model = Model_mHC(CONFIG["D_MODEL"], CONFIG["N_LAYERS"], CONFIG["VOCAB"], CONFIG["STREAMS"], use_mixer=False).cuda()

    optim = torch.optim.AdamW(model.parameters(), lr=CONFIG["LR"])
    loss_fn = nn.CrossEntropyLoss()
    history = {"step": [], "loss": [], "acc": []}
    
    for i in range(CONFIG["STEPS"]):
        optim.zero_grad()
        x, y = get_batch(CONFIG["BS"], CONFIG["SEQ"], CONFIG["VOCAB"])
        logits = model(x)
        
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
            print(f"\rStep {i}: Loss {loss.item():.4f} | Acc {acc.item():.4f}", end="")
            
    return history, None

def log_experiment(run_id, mhc_h, nomix_h): 
    # Create a figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # --- PLOT 1: LOSS ---
    if 'loss' in mhc_h:
        ax1.plot(mhc_h['step'], mhc_h['loss'], label='mHC (With Mixer)', color='blue')
    if 'loss' in nomix_h:
        ax1.plot(nomix_h['step'], nomix_h['loss'], label='No Mixer', color='orange')
    ax1.set_title(f"Loss (Run {run_id})")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()

    # --- PLOT 2: ACCURACY ---
    if 'acc' in mhc_h:
        ax2.plot(mhc_h['step'], mhc_h['acc'], label='mHC (With Mixer)', color='blue')
    if 'acc' in nomix_h:
        ax2.plot(nomix_h['step'], nomix_h['acc'], label='No Mixer', color='orange')
    ax2.set_title(f"Accuracy (Seq {CONFIG['SEQ']})")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    # Save the combined figure
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{run_id}_plot.png"))
    plt.close() # Good practice to close figure to free memory

    # --- LOGGING TO MARKDOWN (Remains the same) ---
    md_path = os.path.join(RESULT_DIR, "experiments_log.md")
    config_str = str(CONFIG).replace(", ", "<br>").replace("{", "").replace("}", "").replace("'", "")
    
    with open(md_path, "a") as f:
        if os.stat(md_path).st_size == 0:
            f.write("| Run ID | Blue Acc | Orange Acc | Config |\n")
            f.write("|---|---|---|---|\n")
        
        blue_acc = mhc_h['acc'][-1] if 'acc' in mhc_h and mhc_h['acc'] else 0.0
        orange_acc = nomix_h['acc'][-1] if 'acc' in nomix_h and nomix_h['acc'] else 0.0
        
        f.write(f"| {run_id} | {blue_acc:.4f} | {orange_acc:.4f} | {config_str} |\n")
if __name__ == "__main__":
    run_id = datetime.datetime.now().strftime("ablation_%Y%m%d_%H%M%S")
    print(f"ID: {run_id} | Config: {CONFIG}")
    
    # 1. Run mHC (Baseline for this ablation)
    mhc_h = run_experiment("mhc", run_id)
    
    # 2. Run No-Mixer (The Test)
    nomix_h = run_experiment("mhc_nomixer", run_id)
    
    log_experiment(run_id, mhc_h, nomix_h)