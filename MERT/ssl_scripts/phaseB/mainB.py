import torch
import argparse
import numpy as np
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, TensorDataset

# Importing from your other cleaned files
from models import MERModel
from losses import SupCRLoss
from data_utils import load_pmemo_data

def train_one_epoch(model, loader, optimizer, mse_crit, supcr_crit, device, use_supcr):
    model.train()
    epoch_mse = 0
    for b_x, b_y in loader:
        b_x, b_y = b_x.to(device), b_y.to(device)
        optimizer.zero_grad()
        preds, latent = model(b_x)
        
        loss_mse = mse_crit(preds, b_y)
        loss = loss_mse
        
        if use_supcr:
            loss_con = supcr_crit(latent, b_y)
            loss = loss_mse + (0.1 * loss_con)
            
        loss.backward()
        optimizer.step()
        epoch_mse += loss_mse.item()
    return epoch_mse / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for b_x, b_y in loader:
            p, _ = model(b_x.to(device))
            all_p.append(p.cpu().numpy())
            all_y.append(b_y.numpy())
    
    y_true = np.vstack(all_y)
    y_pred = np.vstack(all_p)
    return r2_score(y_true, y_pred, multioutput='raw_values')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['baseline', 'hybrid'], default='hybrid')
    parser.add_argument('--mode', choices=['simple', 'kfold'], default='kfold')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()

    # --- 1. Load Data ---
    # Update this path if your CSV is moved!
    csv_path = '/datasets/emotions/PMEmo2019/annotations/static_annotations.csv'
    feature_path = "pmemo_mert_all_layers.pt"
    
    print(f"🚀 Starting Experiment: Model={args.model}, Mode={args.mode}")
    print("📂 Loading data (this might take a few seconds)...")
    
    if not os.path.exists(feature_path):
        print(f"❌ Error: {feature_path} not found in this folder!")
        exit()

    X, Y = load_pmemo_data(feature_path, csv_path)
    print(f"✅ Data Loaded. Total samples: {len(X)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    # --- 2. Run Experiment ---
    if args.mode == 'simple':
        print("🏃 Running Simple Train/Test Split (80/20)...")
        xtr, xte, ytr, yte = train_test_split(X, Y, test_size=0.2, random_state=42)
        train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=32, shuffle=True)
        test_loader = DataLoader(TensorDataset(xte, yte), batch_size=32)
        
        model = MERModel(mode=args.model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        mse_crit = torch.nn.MSELoss()
        supcr_crit = SupCRLoss()

        for epoch in range(args.epochs):
            avg_loss = train_one_epoch(model, train_loader, optimizer, mse_crit, supcr_crit, device, args.model == 'hybrid')
            if epoch % 10 == 0:
                print(f"   Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.4f}")

        scores = evaluate(model, test_loader, device)
        print(f"\n✨ FINAL RESULTS ({args.model})")
        print(f"Arousal R²: {scores[0]:.4f}")
        print(f"Valence R²: {scores[1]:.4f}")

    else:
        print("🧪 Running 5-Fold Cross Validation...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_fold_scores = []
        
        for fold, (tr_idx, te_idx) in enumerate(kf.split(X)):
            print(f"  > Fold {fold+1}/5 training...")
            train_loader = DataLoader(TensorDataset(X[tr_idx], Y[tr_idx]), batch_size=32, shuffle=True)
            test_loader = DataLoader(TensorDataset(X[te_idx], Y[te_idx]), batch_size=32)
            
            model = MERModel(mode=args.model).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2)
            
            for _ in range(args.epochs):
                train_one_epoch(model, train_loader, optimizer, torch.nn.MSELoss(), 
                                SupCRLoss(), device, args.model == 'hybrid')
            
            fold_score = evaluate(model, test_loader, device)
            all_fold_scores.append(fold_score)
            print(f"    Fold {fold+1} Score - A: {fold_score[0]:.3f}, V: {fold_score[1]:.3f}")
        
        avg = np.mean(all_fold_scores, axis=0)
        print(f"\n🏆 K-FOLD AVERAGE ({args.model})")
        print(f"Mean Arousal R²: {avg[0]:.4f}")
        print(f"Mean Valence R²: {avg[1]:.4f}")