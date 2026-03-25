import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import classification_report, precision_recall_fscore_support

# ======================== MODEL ARCHITECTURE ========================

NCLS = 95
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

class QoIAttnBackbone(nn.Module):
    def __init__(self, n_bands=285, d=192, n_heads=4, attn_tau=1.0, use_derivatives=False):
        super().__init__()
        self.n_bands = n_bands
        self.d = d
        self.n_heads = n_heads
        self.tau = attn_tau
        self.use_derivatives = use_derivatives

        dh = d // 2
        in_ch = 3 if use_derivatives else 1
        self.val_fc   = nn.Linear(in_ch, dh)
        self.band_emb = nn.Embedding(n_bands, dh)
        self.proj_in  = nn.Linear(dh * 2, d)

        self.q  = nn.Parameter(torch.randn(n_heads, d))
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)

        self.ln = nn.LayerNorm(d)
        self.scale = nn.Parameter(torch.ones(n_bands))
        self.bias  = nn.Parameter(torch.zeros(n_bands))
        self.attn_bias = nn.Parameter(torch.zeros(n_heads, n_bands))

    def forward(self, x, return_attn=False):
        B, P = x.shape
        x = x * self.scale + self.bias

        if self.use_derivatives:
            dx = torch.zeros_like(x)
            dx[:, 1:] = x[:, 1:] - x[:, :-1]
            ddx = torch.zeros_like(x)
            ddx[:, 2:] = dx[:, 2:] - dx[:, 1:-1]
            feats = torch.stack([x, dx, ddx], dim=-1)
        else:
            feats = x.unsqueeze(-1)

        band_ids = torch.arange(P, device=x.device).unsqueeze(0).expand(B, P)
        tok = torch.cat([self.val_fc(feats), self.band_emb(band_ids)], dim=-1)
        tok = self.proj_in(tok)
        tok = self.ln(tok)

        K = self.Wk(tok)
        V = self.Wv(tok)

        outs, attns = [], []
        scale = (K.size(-1) ** 0.5)
        for h in range(self.n_heads):
            qh = self.q[h].unsqueeze(0).unsqueeze(1)
            scores = (K * qh).sum(-1) / scale
            scores = scores + self.attn_bias[h].unsqueeze(0)
            a = F.softmax(scores / self.tau, dim=1)
            z = (a.unsqueeze(-1) * V).sum(1)
            outs.append(z)
            attns.append(a)

        zcat = torch.cat(outs, dim=-1)
        if return_attn:
            A = torch.stack(attns, dim=1).mean(1)
            return zcat, A
        return zcat

class MineralModel(nn.Module):
    def __init__(self, d=192, n_heads=4, attn_tau=1.0, dropout=0.1, use_derivatives=False, n_bands=285):
        super().__init__()
        self.bb   = QoIAttnBackbone(n_bands=n_bands, d=d, n_heads=n_heads, attn_tau=attn_tau, use_derivatives=use_derivatives)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d * n_heads, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, NCLS)
        )

    def forward(self, x, return_attn=False):
        if return_attn:
            z, A = self.bb(x, return_attn=True)
            logits = self.head(z)
            return logits, A
        z = self.bb(x, return_attn=False)
        return self.head(z)

# ======================== INFERENCE ========================

def load_checkpoint(ckpt_path, device=DEVICE):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    med, iqr = ckpt["median"], ckpt["iqr"]
    model = MineralModel(
        d=ckpt["d_model"], n_heads=ckpt["heads"], attn_tau=ckpt["attn_tau"],
        dropout=0.1, use_derivatives=ckpt["use_derivatives"], n_bands=med.shape[0]
    ).to(device)
    model.bb.load_state_dict(ckpt["backbone"])
    model.head.load_state_dict(ckpt["head"])
    model.eval()
    return model, med, iqr, med.shape[0]

@torch.no_grad()
def predict(model, X_raw, med, iqr, batch_size=1024, return_attention=False, device=DEVICE, top_k=3):
    X_raw = np.asarray(X_raw, dtype=np.float32)
    n_samples, n_bands = X_raw.shape[0], med.shape[0]

    # Auto-assign 0 if last column is 0
    last_col_idx = X_raw.shape[1] - 1
    skip_mask = (X_raw[:, last_col_idx] == 0)
    process_mask = ~skip_mask
    
    top_preds = np.zeros((n_samples, top_k), dtype=np.int32)
    top_confs = np.zeros((n_samples, top_k), dtype=np.float32)
    attention = np.zeros((n_samples, n_bands), dtype=np.float32) if return_attention else None

    top_preds[skip_mask, :] = 0
    top_confs[skip_mask, 0] = 1.0

    n_to_process = np.sum(process_mask)
    if n_to_process > 0:
        X_spectral = X_raw[process_mask, :n_bands]
        process_indices = np.where(process_mask)[0]
        X_norm = (X_spectral - med) / (iqr + 1e-6)
        X_tensor = torch.from_numpy(X_norm).float().to(device)

        for batch_idx in range(0, n_to_process, batch_size):
            end_idx = min(batch_idx + batch_size, n_to_process)
            batch = X_tensor[batch_idx:end_idx]
            orig_indices = process_indices[batch_idx:end_idx]
            
            if return_attention:
                logits, A = model(batch, return_attn=True)
                attention[orig_indices] = A.cpu().numpy()
            else:
                logits = model(batch)
            
            probs = F.softmax(logits, dim=1)
            tk_c, tk_p = torch.topk(probs, k=top_k, dim=1)
            top_preds[orig_indices] = (tk_p.cpu().numpy() + 1)
            top_confs[orig_indices] = tk_c.cpu().numpy()

    return top_preds, top_confs, attention

def main():
    parser = argparse.ArgumentParser(description="Inference with Top-1 and Top-3 Quality Metrics")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Input .npy file")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size")
    parser.add_argument("--save-attn", action="store_true", help="Save attention weights")
    parser.add_argument("--output", type=str, default="predictions.npy", help="Output path")
    args = parser.parse_args()
    start_time = time.time()

    # 1. Load Model and Data
    model, med, iqr, n_bands = load_checkpoint(args.ckpt, device=DEVICE)
    X_new = np.load(args.data)
    
    # Extract Ground Truth (Last column)
    # Ensuring it's integer for comparison
    y_true = X_new[:, -1].astype(np.int32) 
    
    # 2. Run Inference
    # predict() returns top_preds (N, 3), top_confs (N, 3), and attn (N, n_bands)
    top_preds, top_confs, attn = predict(
        model, X_new, med, iqr, 
        batch_size=args.batch, 
        return_attention=args.save_attn, 
        device=DEVICE, 
        top_k=3
    )
    
    # 3. Calculate Quality Metrics
    print("\n" + "="*55)
    print("      MINERAL CLASSIFICATION PERFORMANCE REPORT")
    print("="*55)

    # Filter out background (ID 0) for a clean mineral evaluation
    eval_mask = (y_true > 0)
    
    if np.any(eval_mask):
        y_t = y_true[eval_mask]
        y_p1 = top_preds[eval_mask, 0]  # Standard Top-1 prediction
        
        # --- Top-1 Metrics (Standard) ---
        p1, r1, f1_1, _ = precision_recall_fscore_support(
            y_t, y_p1, average='macro', zero_division=0
        )
        acc1 = np.mean(y_p1 == y_t)

        # --- Top-3 Metrics (Manual Macro Calculation) ---
        # Overall Top-3 Accuracy (Micro)
        top3_hits_mask = np.any(top_preds[eval_mask, :3] == y_t[:, None], axis=1)
        acc3 = np.mean(top3_hits_mask)

        # Macro Top-3 Recall (Average of recall per class)
        unique_classes = np.unique(y_t)
        cls_recalls_top3 = []
        for c in unique_classes:
            c_mask = (y_t == c)
            # Check if class 'c' exists in any of the top 3 columns for these rows
            c_hits = np.any(top_preds[eval_mask][c_mask, :3] == c, axis=1)
            cls_recalls_top3.append(np.mean(c_hits))
        
        macro_r3 = np.mean(cls_recalls_top3)

        # --- Display Results ---
        print(f"{'METRIC':<20} | {'TOP-1':<12} | {'TOP-3':<12}")
        print("-" * 55)
        print(f"{'Accuracy (Micro)':<20} | {acc1:<12.4f} | {acc3:<12.4f}")
        print(f"{'Macro Recall':<20} | {r1:<12.4f} | {macro_r3:<12.4f}")
        print(f"{'Macro Precision':<20} | {p1:<12.4f} | {'N/A':<12}")
        print(f"{'Macro F1-Score':<20} | {f1_1:<12.4f} | {'N/A':<12}")
        print("-" * 55)
        print("Note: Top-3 Macro Recall indicates if the correct mineral")
        print("was among the model's 3 most confident guesses.")

        print("\nDetailed Top-1 Per-Class Report:")
        print(classification_report(y_t, y_p1, zero_division=0))
    else:
        print("Warning: No ground truth (IDs > 0) found in last column. Metrics skipped.")

    # 4. Format Results for Saving [ID1, Conf1, ID2, Conf2, ID3, Conf3]
    results = np.zeros((len(X_new), 6), dtype=np.float32)
    for i in range(3):
        results[:, i*2] = top_preds[:, i]
        results[:, i*2+1] = top_confs[:, i]
    
    np.save(args.output, results)
    print(f"\n✓ Saved top 3 predictions and confidences to: {args.output}")

    # Optional: Save Attention
    if args.save_attn and attn is not None:
        attn_path = args.output.replace(".npy", "") + "_attention.npy"
        np.save(attn_path, attn)
        print(f"✓ Saved attention weights to: {attn_path}")

    elapsed = time.time() - start_time
    print(f"\n[TIMER] Total inference time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

if __name__ == "__main__":
    main()