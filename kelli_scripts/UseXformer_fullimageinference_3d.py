import os
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

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
def predict(model, X_raw, med, iqr, batch_size=1024, return_attention=False, device=DEVICE, top_k=3, gt_array=None, conf_threshold=0.0):
    X_raw = np.asarray(X_raw, dtype=np.float32)
    n_samples, n_bands = X_raw.shape[0], med.shape[0]

    if gt_array is not None:
        skip_mask = (gt_array == 0)
        process_mask = ~skip_mask
    else:
        skip_mask = np.zeros(n_samples, dtype=bool)
        process_mask = np.ones(n_samples, dtype=bool)
    
    top_preds = np.zeros((n_samples, top_k), dtype=np.int32)
    top_confs = np.zeros((n_samples, top_k), dtype=np.float32)
    max_confs = np.zeros(n_samples, dtype=np.float32) 
    attention = np.zeros((n_samples, n_bands), dtype=np.float32) if return_attention else None
    
    # Pre-fill background
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
            
            max_probs = probs.max(dim=1).values.cpu().numpy()
            
            # Convert 0..94 -> 1..95
            batch_preds = (tk_p.cpu().numpy() + 1)
            batch_confs = tk_c.cpu().numpy()
            
            if conf_threshold > 0.0:
                below_threshold = max_probs < conf_threshold
                batch_preds[below_threshold] = 0
                # Keep confidence for debugging even if zeroed
            
            top_preds[orig_indices] = batch_preds
            top_confs[orig_indices] = batch_confs

    return top_preds, top_confs, attention

def main():
    parser = argparse.ArgumentParser(description="Inference with Attention and Threshold Reporting")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Spectral .npy file")
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--save-attn", action="store_true")
    parser.add_argument("--output", type=str, default="predictions.npy")
    parser.add_argument("--no-gt", action="store_true", help="Classify all pixels")
    parser.add_argument("--gt-file", type=str, default=None, help="Path to Ground Truth .npy")
    parser.add_argument("--gt-row", type=int, default=None, help="Row index in GT file")
    parser.add_argument("--conf-threshold", type=float, default=0.0)
    args = parser.parse_args()
    start_time = time.time()

    # 1. Load Model and Data
    model, med, iqr, n_bands_expected = load_checkpoint(args.ckpt, device=DEVICE)
    X_raw = np.load(args.data)

    # Extract scene ID from data filename (e.g., "20230327T212606_2308614_013" from the path)
    import re
    data_stem = Path(args.data).stem
    scene_id_match = re.search(r'(\d{8}T\d{6}_\d{7}_\d{3})', data_stem)
    scene_id = scene_id_match.group(1) if scene_id_match else data_stem

    actual_bands_in_file = X_raw.shape[-1]
    if actual_bands_in_file < n_bands_expected:
        raise ValueError(f"CRITICAL ERROR: File has {actual_bands_in_file} bands, model needs {n_bands_expected}.")

    X_new = X_raw.reshape(-1, actual_bands_in_file) if X_raw.ndim == 3 else X_raw
    
    # 2. Load Ground Truth
    y_true = None
    if not args.no_gt:
        if args.gt_file is None or args.gt_row is None:
            raise ValueError("Must provide --gt-file and --gt-row unless using --no-gt.")
        gt_full = np.load(args.gt_file)
        y_true = gt_full[args.gt_row].astype(np.float32)
        y_true[np.isnan(y_true)] = 0
        y_true = y_true.astype(np.int32).flatten()
        print(f"Loaded GT Row {args.gt_row}. Minerals present: {np.sum(y_true > 0):,} pixels.")

    # 3. Run Inference
    top_preds, top_confs, attn = predict(
        model, X_new, med, iqr, 
        batch_size=args.batch, 
        return_attention=args.save_attn, 
        device=DEVICE, 
        top_k=3,
        gt_array=y_true, 
        conf_threshold=args.conf_threshold
    )

    # 4. Threshold & Confidence Report
    if args.conf_threshold > 0.0:
        total_pix = len(top_preds)
        # Count pixels where prediction became 0 specifically due to threshold
        if args.no_gt:
            zeroed = np.sum(top_preds[:, 0] == 0)
            print(f"Threshold Report: {zeroed:,} / {total_pix:,} pixels fell below {args.conf_threshold} and were set to 0.")
        else:
            mineral_mask = (y_true > 0)
            dropped = np.sum((top_preds[mineral_mask, 0] == 0))
            print(f"Threshold Report: {dropped:,} mineral pixels fell below {args.conf_threshold} and were set to 0.")

    # 5. Accuracy Report
    if y_true is not None:
        eval_mask = (y_true > 0)
        if np.any(eval_mask):
            y_t = y_true[eval_mask]
            y_p1 = top_preds[eval_mask, 0]
            
            p1, r1, f1, _ = precision_recall_fscore_support(y_t, y_p1, average='macro', zero_division=0)
            acc1 = np.mean(y_p1 == y_t)
            
            is_top3 = np.any(top_preds[eval_mask, :3] == y_t[:, None], axis=1)
            acc3 = np.mean(is_top3)
            y_p3_best = np.where(is_top3, y_t, top_preds[eval_mask, 0])
            _, r3, _, _ = precision_recall_fscore_support(y_t, y_p3_best, average='macro', zero_division=0)

            print("\n" + "="*55)
            print(f"MINERAL CLASSIFICATION PERFORMANCE REPORT {Path(args.data).stem}")
            print("="*55)
            print(f"{'METRIC':<20} | {'TOP-1':<12} | {'TOP-3':<12}")
            print("-" * 55)
            print(f"{'Accuracy (Micro)':<20} | {acc1:<12.4f} | {acc3:<12.4f}")
            print(f"{'Macro Recall':<20} | {r1:<12.4f} | {r3:<12.4f}")
            print(f"{'Macro Precision':<20} | {p1:<12.4f} | {'N/A':<12}")
            print(f"{'Macro F1-Score':<20} | {f1:<12.4f} | {'N/A':<12}")
            print("-" * 55)

    # 6. Save Results — append scene ID to output filename
    results = np.zeros((len(X_new), 6), dtype=np.float32)
    for i in range(3):
        results[:, i*2] = top_preds[:, i]
        results[:, i*2+1] = top_confs[:, i]

    out_path = Path(args.output)
    out_file = out_path.with_stem(f"{out_path.stem}_{scene_id}")
    np.save(out_file, results)
    print(f"\n✓ Saved 6-column predictions to: {out_file}")

    if args.save_attn and attn is not None:
        attn_file = out_file.with_stem(f"{out_file.stem}_attention")
        np.save(attn_file, attn)
        print(f"✓ Saved attention weights to: {attn_file} (Shape: {attn.shape})")

    elapsed = time.time() - start_time
    print(f"\n[TIMER] Total inference time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

if __name__ == "__main__":
    main()