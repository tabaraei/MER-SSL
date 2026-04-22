import torch
import pandas as pd
from torch.utils.data import TensorDataset

def load_pmemo_data(feature_path, label_path):
    # Load extracted MERT layers [cite: 14]
    data_dict = torch.load(feature_path, weights_only=False)
    labels_df = pd.read_csv(label_path)

    X_list, Y_list = [], []
    for m_id, layers in data_dict.items():
        row = labels_df[labels_df['musicId'] == int(m_id)]
        if not row.empty:
            X_list.append(torch.from_numpy(layers).float())
            Y_list.append(torch.tensor([row['Arousal(mean)'].values[0], 
                                      row['Valence(mean)'].values[0]]).float())
    
    return torch.stack(X_list), torch.stack(Y_list)