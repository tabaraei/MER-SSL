from sklearn.model_selection import KFold
import numpy as np

# ... (Keep your model and loss imports)

# 1. Prepare your data tensors (X and Y) as before
# X: [794, 25, 1024], Y: [794, 2]

kf = KFold(n_n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    print(f"\n--- Training Fold {fold+1}/5 ---")
    
    # Split data
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=32, shuffle=True)
    
    # Initialize a fresh model for each fold
    model = HybridMERModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # [Insert your training loop here for ~30-50 epochs]
    
    # Evaluation on the Test Fold (Data the model hasn't seen)
    model.eval()
    with torch.no_grad():
        preds, _ = model(X_test.to(device))
        score = r2_score(Y_test.numpy(), preds.cpu().numpy(), multioutput='raw_values')
        fold_results.append(score)
        print(f"Fold {fold+1} Test R2 - Arousal: {score[0]:.3f}, Valence: {score[1]:.3f}")

# Final Scientific Result
avg_scores = np.mean(fold_results, axis=0)
print(f"\nFinal K-Fold Average R2 -> Arousal: {avg_scores[0]:.3f}, Valence: {avg_scores[1]:.3f}")