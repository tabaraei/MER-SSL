import torch
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from models import HybridMERModel

def validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize and Load the Hybrid Model
    model = HybridMERModel().to(device)
    try:
        model.load_state_dict(torch.load("hybrid_mer_model_v1.pth", map_location=device))
        print("Successfully loaded Hybrid Model weights.")
    except FileNotFoundError:
        print("Error: hybrid_mer_model_v1.pth not found. Please train the model first.")
        return

    model.eval()

    # 2. Load Data
    # Fixed the security warning with weights_only=False for custom dicts
    data_dict = torch.load("pmemo_mert_all_layers.pt", weights_only=False)
    labels_df = pd.read_csv('/datasets/emotions/PMEmo2019/annotations/static_annotations.csv')

    all_preds = []
    all_labels = []

    print("Evaluating model on PMEmo dataset...")
    with torch.no_grad():
        for music_id, layers in data_dict.items():
            row = labels_df[labels_df['musicId'] == int(music_id)]
            if not row.empty:
                # Prepare input: [1, 25, 1024]
                X = torch.from_numpy(layers).float().unsqueeze(0).to(device)
                
                # Forward pass
                preds, _ = model(X)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append([
                    row['Arousal(mean)'].values[0], 
                    row['Valence(mean)'].values[0]
                ])

    all_preds = np.vstack(all_preds)
    all_labels = np.array(all_labels)

    # 3. Calculate Scores
    r2_arousal = r2_score(all_labels[:, 0], all_preds[:, 0])
    r2_valence = r2_score(all_labels[:, 1], all_preds[:, 1])

    # 4. Generate Reporting Table
    print("\n" + "="*40)
    print("   HYBRID MODEL PERFORMANCE RESULTS")
    print("="*40)
    print(f"Arousal R²: {r2_arousal:.4f}")
    print(f"Valence R²: {r2_valence:.4f}")
    print("="*40)
    
    print("\n### Comparison for Thesis Report ###")
    print("| Metric | Baseline (Layer 24) | Hybrid (Fusion + SupCR) | Δ Improvement |")
    print("| :--- | :--- | :--- | :--- |")
    # Baseline values from your preliminary report
    print(f"| Arousal $R^2$ | 0.702 | {r2_arousal:.3f} | {r2_arousal - 0.702:+.3f} |")
    print(f"| Valence $R^2$ | 0.515 | {r2_valence:.3f} | {r2_valence - 0.515:+.3f} |")

if __name__ == "__main__":
    validate()