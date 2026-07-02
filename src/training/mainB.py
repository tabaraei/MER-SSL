"""
mainB.py — Master Training Script for MER Phase B
===================================================
Optimized for Thesis Validation:
1. Differential Optimizer: Fixed frozen fusion weights (Entropy 3.2189).
2. Balanced Sampler: Corrected Simpson's Paradox in quadrant analysis.
3. Multimodal Support: Handles Audio + EDA parameter nesting.
"""

from configs.config import PATHS, PHASE_B  # centralised config
import torch
import argparse
import numpy as np
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from models.models import (
    MERModel,
    DualSSLModel,
    DualSSLBottleneckModel,
    DualSSLDomainModel,
    analyze_layer_weights,
    plot_layer_weights,
    analyze_dual_layer_weights,
)
from losses.losses import HybridLoss, CCCLoss, SupCRLoss, fusion_entropy_loss
from utils.data_utils import (
    load_pmemo_data,
    load_pmemo_with_eda,
    load_pmemo_dual_ssl,
    load_iadse_dual_ssl,
    quadrant_r2_breakdown,
    add_quadrant_labels,
    get_emotion_quadrant, # Ensure this is imported for the sampler
)

# =============================================================================
# Helper Functions
# =============================================================================

def get_balanced_sampler(Y):
    """Fixes quadrant imbalance by weighting samples inversely to their frequency."""
    quads = [get_emotion_quadrant(a, v) for a, v in Y.numpy()]
    class_counts = np.bincount(quads, minlength=4)
    # Weights are 1/count; 1e-6 prevents division by zero
    weights = 1. / (class_counts + 1e-6)
    sample_weights = torch.tensor([weights[q] for q in quads])
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def get_optimizer(model, use_eda, base_lr, use_dual=False, use_joint=False, use_bottleneck=False):
    """Provides a high LR for Fusion parameters and base LR for the head."""
    if use_joint:
        # DualSSLDomainModel: both fusions fast; head/regressor/domain_emb at base LR
        params = [
            {'params': model.fusion_mert.parameters(), 'lr': 1e-2},
            {'params': model.fusion_w2v.parameters(),  'lr': 1e-2},
            {'params': model.head.parameters(),        'lr': base_lr},
            {'params': model.regressor.parameters(),   'lr': base_lr},
            {'params': model.domain_emb.parameters(),  'lr': base_lr},
        ]
        return torch.optim.Adam(params, weight_decay=1e-3)
    if use_bottleneck:
        # DualSSLBottleneckModel: fusions + per-encoder projections train fast
        params = [
            {'params': model.fusion_mert.parameters(), 'lr': 1e-2},
            {'params': model.fusion_w2v.parameters(),  'lr': 1e-2},
            {'params': model.proj_mert.parameters(),   'lr': 1e-2},
            {'params': model.proj_w2v.parameters(),    'lr': 1e-2},
            {'params': model.head.parameters(),        'lr': base_lr},
            {'params': model.regressor.parameters(),   'lr': base_lr},
        ]
        return torch.optim.Adam(params, weight_decay=1e-3)
    if use_dual:
        # DualSSLModel: both layer-fusion modules train fast, head/regressor at base LR
        params = [
            {'params': model.fusion_mert.parameters(), 'lr': 1e-2},
            {'params': model.fusion_w2v.parameters(),  'lr': 1e-2},
            {'params': model.head.parameters(),        'lr': base_lr},
            {'params': model.regressor.parameters(),   'lr': base_lr},
        ]
        return torch.optim.Adam(params, weight_decay=1e-3)
    if use_eda:
        # For MERModelWithEDA, parameters are nested in model.base_model
        params = [
            {'params': model.base_model.fusion.parameters(), 'lr': 1e-2},
            {'params': model.base_model.head.parameters(), 'lr': base_lr},
            {'params': model.base_model.regressor.parameters(), 'lr': base_lr},
            {'params': model.eda_proj.parameters(), 'lr': base_lr},
            {'params': model.fusion_head.parameters(), 'lr': base_lr}
        ]
    else:
        # Standard MERModel
        params = [
            {'params': model.fusion.parameters(), 'lr': 1e-2},
            {'params': model.head.parameters(), 'lr': base_lr},
            {'params': model.regressor.parameters(), 'lr': base_lr}
        ]
    return torch.optim.Adam(params, weight_decay=1e-3)

def build_loader(X_a, X_e, Y_t, use_eda, batch_size=32, use_sampler=False, use_dual=False):
    """Universal data loader supporting optional balanced sampling.

    In dual-SSL mode X_a=X_mert and X_e=X_w2v (same 3-tensor layout as EDA).
    """
    ds = TensorDataset(X_a, X_e, Y_t) if (use_eda or use_dual) else TensorDataset(X_a, Y_t)
    
    if use_sampler:
        sampler = get_balanced_sampler(Y_t)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

# =============================================================================
# Training & Evaluation Functions
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device, use_eda=False, use_dual=False, beta=0.0):
    model.train()
    accum = {}
    for batch in loader:
        optimizer.zero_grad()
        if use_eda or use_dual:
            b_a, b_e, b_y = [t.to(device) for t in batch]
            preds, latent = model(b_a, b_e)
        else:
            b_x, b_y = [t.to(device) for t in batch]
            preds, latent = model(b_x)

        loss, components = criterion(preds, latent, b_y)
        if beta > 0.0:
            e_loss = fusion_entropy_loss(model) * beta
            loss = loss + e_loss
            components["loss_entropy"] = e_loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in components.items():
            accum[k] = accum.get(k, 0.0) + v
    return {k: v / len(loader) for k, v in accum.items()}

def evaluate(model, loader, device, use_eda=False, use_dual=False):
    model.eval()
    all_p, all_y = [], []
    ccc_fn = CCCLoss()
    with torch.no_grad():
        for batch in loader:
            if use_eda or use_dual:
                b_a, b_e, b_y = [t.to(device) for t in batch]
                p, _ = model(b_a, b_e)
            else:
                b_x, b_y = [t.to(device) for t in batch]
                p, _ = model(b_x)
            all_p.append(p.cpu()); all_y.append(b_y.cpu())
    
    y_true, y_pred = np.vstack(all_y), np.vstack(all_p)
    ccc_scores = ccc_fn.compute_ccc_scores(torch.tensor(y_pred), torch.tensor(y_true))
    return y_true, y_pred, ccc_scores

def print_results(label, y_true, y_pred, ccc_scores):
    r2 = r2_score(y_true, y_pred, multioutput="raw_values")
    print(f"\n{'='*55}\n  {label}\n{'='*55}")
    print(f"  R²  Arousal : {r2[0]:.4f} | Valence : {r2[1]:.4f}")
    print(f"  CCC Arousal : {ccc_scores['CCC_Arousal']:.4f} | CCC Valence : {ccc_scores['CCC_Valence']:.4f}")
    print(f"\n  Per-Quadrant R² Breakdown:")
    breakdown = quadrant_r2_breakdown(y_true, y_pred)
    for q, s in breakdown.items():
        if s["R2_Arousal"] is not None:
            print(f"    {q}: A={s['R2_Arousal']:.3f} | V={s['R2_Valence']:.3f} | n={s['n']}")
    print(f"{'='*55}")
    return r2

# =============================================================================
# Joint Learning (dual_joint): PMEmo music + IADS-E generalized sound
# =============================================================================

def get_joint_sampler(Y_music, n_iadse):
    """Balanced sampler over a [music ; iadse] concatenated train set.
    Music samples get inverse-quadrant-frequency weight (Simpson's-paradox fix
    on the EVALUATED domain only); IADS-E samples get uniform 1/N_iadse so the
    sounds augment without dominating the gradient."""
    quads = [get_emotion_quadrant(a, v) for a, v in Y_music.numpy()]
    counts = np.bincount(quads, minlength=4)
    w_music = 1. / (counts + 1e-6)
    music_w = [float(w_music[q]) for q in quads]
    iadse_w = [1.0 / max(n_iadse, 1)] * n_iadse
    sample_weights = torch.tensor(music_w + iadse_w, dtype=torch.float)
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def train_one_epoch_joint(model, loader, optimizer, criterion, device):
    model.train()
    accum = {}
    for b_m, b_w, b_y, b_d in loader:
        b_m, b_w, b_y, b_d = b_m.to(device), b_w.to(device), b_y.to(device), b_d.to(device)
        optimizer.zero_grad()
        preds, latent = model(b_m, b_w, b_d)
        # SupCR inside HybridLoss spans the FULL mixed batch → pulls same-quadrant
        # music+sound together in latent space (the core SSL joint-learning insight)
        loss, components = criterion(preds, latent, b_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for k, v in components.items():
            accum[k] = accum.get(k, 0.0) + v
    return {k: v / len(loader) for k, v in accum.items()}

def evaluate_joint(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    ccc_fn = CCCLoss()
    with torch.no_grad():
        for b_m, b_w, b_y, b_d in loader:
            p, _ = model(b_m.to(device), b_w.to(device), b_d.to(device))
            all_p.append(p.cpu()); all_y.append(b_y.cpu())
    y_true, y_pred = np.vstack(all_y), np.vstack(all_p)
    ccc_scores = ccc_fn.compute_ccc_scores(torch.tensor(y_pred), torch.tensor(y_true))
    return y_true, y_pred, ccc_scores

def run_joint(args, device):
    """dual_joint: K-fold on PMEmo ONLY; each fold's train set is augmented with
    a sampled slice of IADS-E (domain=1). Never tests on IADS-E for the headline
    metric (apples-to-apples vs. Step-1 dual)."""
    use_supcr = (args.model == "hybrid")
    criterion = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3,
                           w_supcr=0.1 if use_supcr else 0.0, use_supcr=use_supcr)

    print("\n🌍 Joint Learning | loading PMEmo (domain=0) + IADS-E (domain=1)...")
    Xm_p, Xw_p, Y_p, _ = load_pmemo_dual_ssl(args.feat_path, args.w2v_path, args.csv_path)
    Xm_s, Xw_s, Y_s, _ids_s, _dom_s = load_iadse_dual_ssl(
        args.iadse_mert, args.iadse_w2v, args.iadse_csv,
        args.iadse_val_col, args.iadse_ar_col, args.iadse_id_col,
    )
    n_s = len(Y_s)

    def _qcov(tag, Y):
        q = np.bincount([get_emotion_quadrant(a, v) for a, v in Y.numpy()], minlength=4)
        print(f"    {tag:<12} N={len(Y):<4} | HVHA={q[0]} HVLA={q[1]} LVHA={q[2]} LVLA={q[3]}")
    print("  📐 V-A Quadrant Coverage:")
    _qcov("PMEmo", Y_p); _qcov("IADS-E", Y_s)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    all_y_true, all_y_pred, all_ccc = [], [], []
    last_model = None

    for fold, (tr_idx, te_idx) in enumerate(kf.split(np.arange(len(Y_p)))):
        print(f"  ── Fold {fold+1}/5 ──────────────────────────────")
        rng = np.random.default_rng(42 + fold)

        # Sample IADS-E by k, PMEmo-train by p (reproducible per fold)
        k_n = max(0, min(n_s, int(round(args.mix_k * n_s))))
        s_sel = np.sort(rng.choice(n_s, size=k_n, replace=False)) if k_n < n_s else np.arange(n_s)
        p_n = max(0, min(len(tr_idx), int(round(args.mix_p * len(tr_idx)))))
        p_sel = np.sort(rng.choice(len(tr_idx), size=p_n, replace=False)) if p_n < len(tr_idx) else np.arange(len(tr_idx))
        m_tr = tr_idx[p_sel]

        Xm_tr = torch.cat([Xm_p[m_tr], Xm_s[s_sel]], dim=0)
        Xw_tr = torch.cat([Xw_p[m_tr], Xw_s[s_sel]], dim=0)
        Y_tr  = torch.cat([Y_p[m_tr],  Y_s[s_sel]],  dim=0)
        d_tr  = torch.cat([torch.zeros(len(m_tr), dtype=torch.long),
                           torch.ones(len(s_sel), dtype=torch.long)], dim=0)
        print(f"     train = music({len(m_tr)}) + sound({len(s_sel)})  | test = PMEmo({len(te_idx)})")

        train_loader = DataLoader(
            TensorDataset(Xm_tr, Xw_tr, Y_tr, d_tr),
            batch_size=args.batch_size,
            sampler=get_joint_sampler(Y_p[m_tr], len(s_sel)),
        )
        test_loader = DataLoader(
            TensorDataset(Xm_p[te_idx], Xw_p[te_idx], Y_p[te_idx],
                          torch.zeros(len(te_idx), dtype=torch.long)),
            batch_size=args.batch_size, shuffle=False,
        )

        model = DualSSLDomainModel().to(device)
        optimizer = get_optimizer(model, False, args.lr, use_joint=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        for _ in range(args.epochs):
            train_one_epoch_joint(model, train_loader, optimizer, criterion, device)
            scheduler.step()

        y_t, y_p, ccc = evaluate_joint(model, test_loader, device)
        all_y_true.append(y_t); all_y_pred.append(y_p); all_ccc.append(ccc)
        last_model = model

        if args.eval_on == "both":
            iadse_loader = DataLoader(
                TensorDataset(Xm_s, Xw_s, Y_s, torch.ones(n_s, dtype=torch.long)),
                batch_size=args.batch_size, shuffle=False,
            )
            iy_t, iy_p, iccc = evaluate_joint(model, iadse_loader, device)
            ir2 = r2_score(iy_t, iy_p, multioutput="raw_values")
            print(f"     [IADS-E hold-out] R² A={ir2[0]:.3f} V={ir2[1]:.3f} | "
                  f"CCC A={iccc['CCC_Arousal']:.3f} V={iccc['CCC_Valence']:.3f}")

    all_true, all_pred = np.vstack(all_y_true), np.vstack(all_y_pred)
    avg_ccc_a = np.mean([c["CCC_Arousal"] for c in all_ccc])
    avg_ccc_v = np.mean([c["CCC_Valence"] for c in all_ccc])
    print_results("🌍 JOINT LEARNING FINAL AVERAGE",
                  all_true, all_pred,
                  {"CCC_Arousal": avg_ccc_a, "CCC_Valence": avg_ccc_v})

    torch.save(last_model.state_dict(), "best_model.pt")
    print("💾  Model saved → best_model.pt")
    analyze_dual_layer_weights(last_model, save_dir=".")

# =============================================================================
# EDA-aware MERModel wrapper
# =============================================================================

class MERModelWithEDA(torch.nn.Module):
    def __init__(self, base_model, eda_dim=7):
        super().__init__()
        self.base_model = base_model
        self.eda_proj = torch.nn.Sequential(torch.nn.Linear(eda_dim, 32), torch.nn.ReLU())
        self.fusion_head = torch.nn.Sequential(
            torch.nn.Linear(128 + 32, 64), torch.nn.ReLU(),
            torch.nn.Dropout(0.3), torch.nn.Linear(64, 2)
        )
    def forward(self, x_audio, x_eda):
        _, latent = self.base_model(x_audio)
        eda_feat = self.eda_proj(x_eda)
        return self.fusion_head(torch.cat([latent, eda_feat], dim=1)), latent

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MER Phase B Training")
    parser.add_argument("--model",    choices=["baseline", "hybrid"], default="hybrid")
    parser.add_argument("--mode",     choices=["simple", "kfold"],   default="kfold")
    parser.add_argument("--loss",     choices=["mse", "ccc", "hybrid"], default="hybrid")
    parser.add_argument("--epochs",   type=int, default=PHASE_B.num_epochs)
    parser.add_argument("--lr",       type=float, default=PHASE_B.learning_rate)
    parser.add_argument("--batch_size", type=int, default=PHASE_B.batch_size)
    parser.add_argument("--use_eda",  action="store_true")
    parser.add_argument("--eda_dir",  type=str, default="")
    parser.add_argument("--encoder",  choices=["mert", "dual", "dual_bottleneck", "dual_joint"], default="mert",
                        help="mert = MERT only; dual = MERT+wav2vec2 (concat); "
                             "dual_bottleneck = MERT+wav2vec2 with per-encoder compression (fixes fusion collapse); "
                             "dual_joint = dual + IADS-E joint learning (domain-conditioned)")
    parser.add_argument("--beta", type=float, default=0.01,
                        help="Entropy sharpening penalty weight. Penalizes uniform fusion weights. "
                             "0 = disabled. Recommended range: 0.005–0.05.")
    parser.add_argument("--feat_path", type=str, default=str(PATHS.mert_features))
    parser.add_argument("--w2v_path",  type=str, default=str(PATHS.wav2vec_features))
    parser.add_argument("--csv_path", type=str, default=str(PATHS.pmemo_annotations))
    # --- IADS-E joint-learning args (only used when --encoder dual_joint) ---
    parser.add_argument("--iadse_mert", type=str, default="iadse_mert_all_layers.pt")
    parser.add_argument("--iadse_w2v",  type=str, default="iadse_wav2vec_all_layers.pt")
    parser.add_argument("--iadse_csv",  type=str, default="iadse_labels.csv")
    parser.add_argument("--iadse_val_col", type=str, default="valence")
    parser.add_argument("--iadse_ar_col",  type=str, default="arousal")
    parser.add_argument("--iadse_id_col",  type=str, default="sound_id")
    parser.add_argument("--mix_k", type=float, default=1.0, help="IADS-E ratio kept (Simonetta k)")
    parser.add_argument("--mix_p", type=float, default=1.0, help="PMEmo-train ratio kept (Simonetta p)")
    parser.add_argument("--eval_on", choices=["pmemo", "both"], default="pmemo")
    args = parser.parse_args()

    use_bottleneck = (args.encoder == "dual_bottleneck")
    use_dual  = (args.encoder in ("dual", "dual_bottleneck"))
    use_joint = (args.encoder == "dual_joint")
    if (use_dual or use_joint) and args.use_eda:
        raise SystemExit("❌ --use_eda not yet supported with --encoder dual/dual_bottleneck/dual_joint.")

    print(f"\n🚀 MER Phase B | model={args.model} | mode={args.mode} | encoder={args.encoder} | beta={args.beta}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if use_joint:
        run_joint(args, device)
        raise SystemExit(0)

    # 1. Load Data
    if use_dual:
        X_audio, X_eda, Y, _ = load_pmemo_dual_ssl(args.feat_path, args.w2v_path, args.csv_path)
        use_eda = False
    elif args.use_eda and args.eda_dir:
        X_audio, X_eda, Y = load_pmemo_with_eda(args.feat_path, args.csv_path, args.eda_dir)
        use_eda = True
    else:
        X_audio, Y, _ = load_pmemo_data(args.feat_path, args.csv_path)
        X_eda, use_eda = None, False

    # 2. Configure Loss
    use_supcr = (args.model == "hybrid")
    criterion = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1 if use_supcr else 0.0, use_supcr=use_supcr)

    def build_model():
        if use_bottleneck:
            return DualSSLBottleneckModel().to(device)
        if use_dual:
            return DualSSLModel().to(device)
        base = MERModel(mode=args.model).to(device)
        return MERModelWithEDA(base).to(device) if use_eda else base

    # 3. Execution
    if args.mode == "simple":
        print("\n🏃 Simple 80/20 Train/Test Split (Normal Optimizer)...")
        tr_idx, te_idx = train_test_split(np.arange(len(X_audio)), test_size=0.2, random_state=42)
        
        train_loader = build_loader(X_audio[tr_idx], X_eda[tr_idx] if (use_eda or use_dual) else None, Y[tr_idx], use_eda, args.batch_size, False, use_dual)
        test_loader = build_loader(X_audio[te_idx], X_eda[te_idx] if (use_eda or use_dual) else None, Y[te_idx], use_eda, args.batch_size, False, use_dual)

        model = build_model()
        # "Normal" optimizer for simple mode as requested
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        for epoch in range(args.epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device, use_eda, use_dual, beta=args.beta)
            scheduler.step()

        y_t, y_p, ccc = evaluate(model, test_loader, device, use_eda, use_dual)
        print_results("SIMPLE SPLIT EVALUATION", y_t, y_p, ccc)

        torch.save(model.state_dict(), "best_model.pt")
        print("💾  Model saved → best_model.pt")

    else:
        print("\n🧪 5-Fold Cross-Validation (Balanced & Differential Optimizer)...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_y_true, all_y_pred, all_ccc = [], [], []

        for fold, (tr_idx, te_idx) in enumerate(kf.split(np.arange(len(X_audio)))):
            print(f"  ── Fold {fold+1}/5 ──────────────────────────────")
            train_loader = build_loader(X_audio[tr_idx], X_eda[tr_idx] if (use_eda or use_dual) else None, Y[tr_idx], use_eda, args.batch_size, True, use_dual)
            test_loader = build_loader(X_audio[te_idx], X_eda[te_idx] if (use_eda or use_dual) else None, Y[te_idx], use_eda, args.batch_size, False, use_dual)

            model = build_model()
            # Differential optimizer — fusions/projections at high LR, head at base LR
            optimizer = get_optimizer(model, use_eda, args.lr, use_dual, use_bottleneck=use_bottleneck)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            for _ in range(args.epochs):
                train_one_epoch(model, train_loader, optimizer, criterion, device, use_eda, use_dual, beta=args.beta)
                scheduler.step()

            y_t, y_p, ccc = evaluate(model, test_loader, device, use_eda, use_dual)
            all_y_true.append(y_t); all_y_pred.append(y_p); all_ccc.append(ccc)
            last_model = model

        # 4. Final Aggregation
        all_true, all_pred = np.vstack(all_y_true), np.vstack(all_y_pred)
        avg_ccc_a = np.mean([c["CCC_Arousal"] for c in all_ccc])
        avg_ccc_v = np.mean([c["CCC_Valence"] for c in all_ccc])
        print_results("🏆 K-FOLD FINAL AVERAGE", all_true, all_pred, {"CCC_Arousal": avg_ccc_a, "CCC_Valence": avg_ccc_v})

        torch.save(last_model.state_dict(), "best_model.pt")
        print("💾  Model saved → best_model.pt")

        # 5. Layer Analysis
        if use_dual:
            analyze_dual_layer_weights(last_model, save_dir=".")
        else:
            base_m = model.base_model if use_eda else model
            analyze_layer_weights(base_m, save_path="layer_weights.npy")
            plot_layer_weights(base_m, save_path="layer_weights_kfold.png")
    

# """
# mainB.py — Master Training Script for MER Phase B
# ===================================================
# Runs all experiments needed for thesis:
#   1. Baseline vs Hybrid K-Fold (original, reproduced cleanly)
#   2. CCC + Rank loss variant (new)
#   3. EDA multimodal fusion (new)
#   4. Full evaluation: R², CCC scores, quadrant breakdown
#   5. Layer weight analysis + visualization

# Usage:
#     # Standard hybrid with all new losses (recommended):
#     python mainB.py --model hybrid --mode kfold --loss hybrid

#     # Baseline for comparison:
#     python mainB.py --model baseline --mode kfold --loss mse

#     # With EDA fusion:
#     python mainB.py --model hybrid --mode kfold --loss hybrid --use_eda --eda_dir /path/to/eda

#     # Quick single split for debugging:
#     python mainB.py --model hybrid --mode simple --loss hybrid --epochs 30
# """

# import torch
# import argparse
# import numpy as np
# import os
# from sklearn.model_selection import KFold, train_test_split
# from sklearn.metrics import r2_score
# from torch.utils.data import DataLoader, TensorDataset

# from models import MERModel, analyze_layer_weights, plot_layer_weights
# from losses import HybridLoss, CCCLoss, SupCRLoss
# from data_utils import (
#     load_pmemo_data,
#     load_pmemo_with_eda,
#     quadrant_r2_breakdown,
#     add_quadrant_labels,
# )


# # =============================================================================
# # Training & Evaluation Functions
# # =============================================================================

# def train_one_epoch(
#     model:     MERModel,
#     loader:    DataLoader,
#     optimizer: torch.optim.Optimizer,
#     criterion: HybridLoss,
#     device:    torch.device,
# ) -> dict:
#     """
#     Runs one full training epoch.
#     Returns dict of average loss components for logging.
#     """
#     model.train()
#     accum = {}

#     for b_x, b_y in loader:
#         b_x, b_y = b_x.to(device), b_y.to(device)
#         optimizer.zero_grad()

#         preds, latent = model(b_x)
#         loss, components = criterion(preds, latent, b_y)

#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         for k, v in components.items():
#             accum[k] = accum.get(k, 0.0) + v

#     n = len(loader)
#     return {k: v / n for k, v in accum.items()}


# def train_one_epoch_with_eda(
#     model:     MERModel,
#     loader:    DataLoader,   # yields (x_audio, x_eda, y)
#     optimizer: torch.optim.Optimizer,
#     criterion: HybridLoss,
#     device:    torch.device,
# ) -> dict:
#     model.train()
#     accum = {}

#     for b_audio, b_eda, b_y in loader:
#         b_audio = b_audio.to(device)
#         b_eda   = b_eda.to(device)
#         b_y     = b_y.to(device)
#         optimizer.zero_grad()

#         preds, latent = model(b_audio, b_eda)
#         loss, components = criterion(preds, latent, b_y)

#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         for k, v in components.items():
#             accum[k] = accum.get(k, 0.0) + v

#     n = len(loader)
#     return {k: v / n for k, v in accum.items()}


# def evaluate(
#     model:   MERModel,
#     loader:  DataLoader,
#     device:  torch.device,
#     use_eda: bool = False,
# ) -> tuple[np.ndarray, np.ndarray, dict]:
#     """
#     Evaluates the model on a DataLoader.

#     Returns:
#         y_true      : (N, 2) numpy
#         y_pred      : (N, 2) numpy
#         ccc_scores  : dict with CCC_Arousal and CCC_Valence
#     """
#     model.eval()
#     all_p, all_y = [], []
#     ccc_loss_fn  = CCCLoss()

#     with torch.no_grad():
#         if use_eda:
#             for b_audio, b_eda, b_y in loader:
#                 p, _ = model(b_audio.to(device), b_eda.to(device))
#                 all_p.append(p.cpu())
#                 all_y.append(b_y)
#         else:
#             for b_x, b_y in loader:
#                 p, _ = model(b_x.to(device))
#                 all_p.append(p.cpu())
#                 all_y.append(b_y)

#     y_true = np.vstack([t.numpy() for t in all_y])
#     y_pred = np.vstack([t.numpy() for t in all_p])

#     pred_t   = torch.tensor(y_pred)
#     target_t = torch.tensor(y_true)
#     ccc_scores = ccc_loss_fn.compute_ccc_scores(pred_t, target_t)

#     return y_true, y_pred, ccc_scores


# def print_results(label: str, y_true: np.ndarray, y_pred: np.ndarray, ccc_scores: dict):
#     """Pretty-prints all evaluation metrics."""
#     r2 = r2_score(y_true, y_pred, multioutput="raw_values")
#     print(f"\n{'='*55}")
#     print(f"  {label}")
#     print(f"{'='*55}")
#     print(f"  R²  Arousal : {r2[0]:.4f}")
#     print(f"  R²  Valence : {r2[1]:.4f}")
#     print(f"  CCC Arousal : {ccc_scores['CCC_Arousal']:.4f}")
#     print(f"  CCC Valence : {ccc_scores['CCC_Valence']:.4f}")

#     print(f"\n  Per-Quadrant R² Breakdown:")
#     breakdown = quadrant_r2_breakdown(y_true, y_pred)
#     for q_name, scores in breakdown.items():
#         if scores["R2_Arousal"] is None:
#             print(f"    {q_name}: n={scores['n']} (too few samples)")
#         else:
#             print(f"    {q_name}: A={scores['R2_Arousal']:.3f} | V={scores['R2_Valence']:.3f} | n={scores['n']}")
#     print(f"{'='*55}")
#     return r2


# # =============================================================================
# # EDA-aware MERModel wrapper (late fusion)
# # =============================================================================

# class MERModelWithEDA(torch.nn.Module):
#     """
#     Extends MERModel with late EDA fusion.
#     Concatenates EDA features to the fused audio embedding
#     before the regression head.
#     """
#     def __init__(self, base_model: MERModel, eda_dim: int = 7, hidden_dim: int = 1024):
#         super().__init__()
#         self.base_model = base_model

#         # EDA projection: 7 → 32
#         self.eda_proj = torch.nn.Sequential(
#             torch.nn.Linear(eda_dim, 32),
#             torch.nn.ReLU(),
#         )

#         # Override the regressor to accept fused bottleneck + eda
#         # base bottleneck = 128, eda_proj output = 32 → total = 160
#         self.fusion_head = torch.nn.Sequential(
#             torch.nn.Linear(128 + 32, 64),
#             torch.nn.ReLU(),
#             torch.nn.Dropout(0.3),
#             torch.nn.Linear(64, 2),
#         )

#     def forward(self, x_audio, x_eda):
#         # Get latent from base model (ignores base regressor output)
#         _, latent = self.base_model(x_audio)         # (B, 128)
#         eda_feat  = self.eda_proj(x_eda)              # (B, 32)
#         combined  = torch.cat([latent, eda_feat], dim=1)  # (B, 160)
#         preds     = self.fusion_head(combined)        # (B, 2)
#         return preds, latent   # return latent for SupCR

#     def get_layer_weights(self):
#         return self.base_model.get_layer_weights()


# # =============================================================================
# # Main
# # =============================================================================

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="MER Phase B Training")
#     parser.add_argument("--model",    choices=["baseline", "hybrid"], default="hybrid")
#     parser.add_argument("--mode",     choices=["simple", "kfold"],   default="kfold")
#     parser.add_argument("--loss",     choices=["mse", "ccc", "hybrid"], default="hybrid",
#                         help="mse=original MSE only | ccc=MSE+CCC | hybrid=MSE+CCC+Rank+SupCR")
#     parser.add_argument("--epochs",   type=int, default=PHASE_B.num_epochs)
#     parser.add_argument("--lr",       type=float, default=PHASE_B.learning_rate)
#     parser.add_argument("--use_eda",  action="store_true",
#                         help="Enable EDA multimodal fusion")
#     parser.add_argument("--eda_dir",  type=str, default="",
#                         help="Path to PMEmo EDA directory (required if --use_eda)")
#     parser.add_argument("--csv_path", type=str,
#                         default=str(PATHS.pmemo_annotations))
#     parser.add_argument("--feat_path", type=str,
#                         default=str(PATHS.mert_features))
#     args = parser.parse_args()

#     # -------------------------------------------------------------------------
#     print(f"\n🚀  MER Phase B | model={args.model} | mode={args.mode} | loss={args.loss}")
#     print(f"    EDA fusion: {'ON → ' + args.eda_dir if args.use_eda else 'OFF'}")

#     if not os.path.exists(args.feat_path):
#         print(f"❌ Feature file not found: {args.feat_path}")
#         exit(1)

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"💻  Device: {device}")

#     # -------------------------------------------------------------------------
#     # 1. Load Data
#     # -------------------------------------------------------------------------
#     print("\n📂  Loading data...")
#     if args.use_eda and args.eda_dir:
#         X_audio, X_eda, Y = load_pmemo_with_eda(args.feat_path, args.csv_path, args.eda_dir)
#         use_eda = True
#     else:
#         # Updated to catch the 3rd return value (IDs)
#         X_audio, Y, _ = load_pmemo_data(args.feat_path, args.csv_path)
#         X_eda      = None
#         use_eda    = False
#     # -------------------------------------------------------------------------
#     # 2. Configure Loss
#     # -------------------------------------------------------------------------
#     if args.loss == "mse":
#         criterion = HybridLoss(w_mse=1.0, w_ccc=0.0, w_rank=0.0, w_supcr=0.0, use_supcr=False)
#         print("🔧  Loss: MSE only")
#     elif args.loss == "ccc":
#         criterion = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.0, w_supcr=0.0, use_supcr=False)
#         print("🔧  Loss: MSE + CCC")
#     else:  # hybrid
#         use_supcr = (args.model == "hybrid")
#         criterion = HybridLoss(w_mse=1.0, w_ccc=0.5, w_rank=0.3, w_supcr=0.1, use_supcr=use_supcr)
#         print("🔧  Loss: MSE + CCC + Rank + SupCR" if use_supcr else "🔧  Loss: MSE + CCC + Rank")

#     # -------------------------------------------------------------------------
#     # 3. Helper to build model
#     # -------------------------------------------------------------------------
#     def build_model():
#         base = MERModel(mode=args.model).to(device)
#         if use_eda:
#             return MERModelWithEDA(base).to(device)
#         return base

#     def build_loader(X_a, X_e, Y_t, shuffle=False, batch_size=32):
#         if use_eda:
#             ds = TensorDataset(X_a, X_e, Y_t)
#         else:
#             ds = TensorDataset(X_a, Y_t)
#         return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

#     # -------------------------------------------------------------------------
#     # 4. Run Experiment
#     # -------------------------------------------------------------------------

#     if args.mode == "simple":
#         print("\n🏃  Simple 80/20 Train/Test Split...")

#         idx = np.arange(len(X_audio))
#         tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=42)

#         if use_eda:
#             train_loader = build_loader(X_audio[tr_idx], X_eda[tr_idx], Y[tr_idx], shuffle=True)
#             test_loader  = build_loader(X_audio[te_idx], X_eda[te_idx], Y[te_idx])
#         else:
#             train_loader = build_loader(X_audio[tr_idx], None, Y[tr_idx], shuffle=True)
#             test_loader  = build_loader(X_audio[te_idx], None, Y[te_idx])

#         model     = build_model()
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
#         # Use a higher learning rate specifically for the fusion parameters
#         optimizer = torch.optim.Adam([
#         {'params': model.fusion.parameters(), 'lr': 1e-3}, # Higher LR for weights
#         {'params': model.head.parameters(), 'lr': 2e-4},
#         {'params': model.regressor.parameters(), 'lr': 2e-4}
# ], weight_decay=1e-3)
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

#         for epoch in range(args.epochs):
#             if use_eda:
#                 comp = train_one_epoch_with_eda(model, train_loader, optimizer, criterion, device)
#             else:
#                 comp = train_one_epoch(model, train_loader, optimizer, criterion, device)

#             scheduler.step()
#             if epoch % 5 == 0 or epoch == args.epochs - 1: # Increased frequency
#                 # Using .get() for SupCR in case use_supcr is False
#                 print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
#                       f"MSE={comp['loss_mse']:.4f} | CCC={comp['loss_ccc']:.4f} | "
#                       f"Rank={comp['loss_rank']:.4f} | SupCR={comp.get('loss_supcr', 0):.4f}")

#         y_true, y_pred, ccc_scores = evaluate(model, test_loader, device, use_eda)
#         print_results(f"SIMPLE SPLIT — {args.model.upper()} ({args.loss})", y_true, y_pred, ccc_scores)

#         # Layer weight analysis for hybrid
#         if args.model == "hybrid":
#             analysis = analyze_layer_weights(
#                 model.base_model if use_eda else model
#             )
#             plot_layer_weights(
#                 model.base_model if use_eda else model,
#                 save_path="layer_weights_simple.png"
#             )

#     else:
#         print("\n🧪  5-Fold Cross-Validation...")
#         kf = KFold(n_splits=5, shuffle=True, random_state=42)

#         all_r2       = []
#         all_ccc      = []
#         all_y_true   = []
#         all_y_pred   = []
#         last_model   = None

#         indices = np.arange(len(X_audio))

#         for fold, (tr_idx, te_idx) in enumerate(kf.split(indices)):
#             print(f"\n  ── Fold {fold+1}/5 ──────────────────────────────")

#             if use_eda:
#                 train_loader = build_loader(X_audio[tr_idx], X_eda[tr_idx], Y[tr_idx], shuffle=True)
#                 test_loader  = build_loader(X_audio[te_idx], X_eda[te_idx], Y[te_idx])
#             else:
#                 train_loader = build_loader(X_audio[tr_idx], None, Y[tr_idx], shuffle=True)
#                 test_loader  = build_loader(X_audio[te_idx], None, Y[te_idx])

#             model     = build_model()
#             optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
#             scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

#             for epoch in range(args.epochs):
#                 if use_eda:
#                     train_one_epoch_with_eda(model, train_loader, optimizer, criterion, device)
#                 else:
#                     train_one_epoch(model, train_loader, optimizer, criterion, device)
#                 scheduler.step()

#             y_true, y_pred, ccc_scores = evaluate(model, test_loader, device, use_eda)
#             r2 = r2_score(y_true, y_pred, multioutput="raw_values")

#             print(f"    R²  — Arousal: {r2[0]:.4f} | Valence: {r2[1]:.4f}")
#             print(f"    CCC — Arousal: {ccc_scores['CCC_Arousal']:.4f} | Valence: {ccc_scores['CCC_Valence']:.4f}")

#             all_r2.append(r2)
#             all_ccc.append(ccc_scores)
#             all_y_true.append(y_true)
#             all_y_pred.append(y_pred)
#             last_model = model

#         # ── Aggregate Results ──────────────────────────────────────────────
#         avg_r2  = np.mean(all_r2, axis=0)
#         std_r2  = np.std(all_r2, axis=0)
#         avg_ccc_a = np.mean([s["CCC_Arousal"] for s in all_ccc])
#         avg_ccc_v = np.mean([s["CCC_Valence"] for s in all_ccc])

#         print(f"\n{'='*55}")
#         print(f"  🏆  K-FOLD AVERAGE — {args.model.upper()} ({args.loss})")
#         print(f"{'='*55}")
#         print(f"  R²  Arousal : {avg_r2[0]:.4f} ± {std_r2[0]:.4f}")
#         print(f"  R²  Valence : {avg_r2[1]:.4f} ± {std_r2[1]:.4f}")
#         print(f"  CCC Arousal : {avg_ccc_a:.4f}")
#         print(f"  CCC Valence : {avg_ccc_v:.4f}")

#         # Full quadrant breakdown over all folds combined
#         all_true_combined = np.vstack(all_y_true)
#         all_pred_combined = np.vstack(all_y_pred)
#         print(f"\n  Per-Quadrant Breakdown (all folds combined):")
#         breakdown = quadrant_r2_breakdown(all_true_combined, all_pred_combined)
#         for q_name, scores in breakdown.items():
#             if scores["R2_Arousal"] is None:
#                 print(f"    {q_name}: n={scores['n']} (too few)")
#             else:
#                 print(f"    {q_name}: A={scores['R2_Arousal']:.3f} | V={scores['R2_Valence']:.3f} | n={scores['n']}")
#         print(f"{'='*55}")

#         # ── Layer Weight Analysis (last fold model) ────────────────────────
#         if args.model == "hybrid" and last_model is not None:
#             print("\n📊  Analyzing learned layer weights (final fold)...")
#             base_m = last_model.base_model if use_eda else last_model
#             analyze_layer_weights(base_m, save_path="layer_weights.npy")
#             plot_layer_weights(base_m, save_path="layer_weights_kfold.png")

