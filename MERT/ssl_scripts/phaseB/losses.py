import torch
import torch.nn as nn

class SupCRLoss(nn.Module):
    def __init__(self, temperature=0.07, threshold=0.30):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold 

    def forward(self, features, labels):
        batch_size = features.shape[0]
        
        # similarity matrix (cosine similarity since features are normalized)
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Numeric stability: subtract max to prevent exp(large_number)
        logits_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Mask for positive pairs
        label_dist = torch.cdist(labels, labels, p=2)
        mask = (label_dist < self.threshold).float().to(features.device)
        
        # Remove self-similarity
        logits_mask = torch.scatter(torch.ones_like(mask), 1, 
                                   torch.arange(batch_size).view(-1, 1).to(features.device), 0)
        mask = mask * logits_mask

        # Log-Sum-Exp for denominator
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6) # Add epsilon
        
        # Mean log-probability over positive samples
        pos_per_sample = mask.sum(1)
        # Only calculate loss for samples that actually have a positive neighbor in the batch
        valid_indices = pos_per_sample > 0
        
        if valid_indices.any():
            mean_log_prob_pos = (mask[valid_indices] * log_prob[valid_indices]).sum(1) / pos_per_sample[valid_indices]
            return -mean_log_prob_pos.mean()
        else:
            return torch.tensor(0.0, requires_grad=True).to(features.device)