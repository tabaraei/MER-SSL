"""
losses.py — Complete Loss Functions for MER Hybrid Model
=========================================================
Contains:
  - SupCRLoss       : Supervised Contrastive Regression Loss (original)
  - CCCLoss         : Concordance Correlation Coefficient Loss (NEW)
  - RankLoss        : Differentiable Spearman-proxy Ranking Loss (NEW)
  - HybridLoss      : Combined MSE + CCC + Rank + SupCR (NEW, drop-in for training)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. ORIGINAL: Supervised Contrastive Regression Loss
# =============================================================================

class SupCRLoss(nn.Module):
    """
    Supervised Contrastive Regression Loss.
    Pulls together embeddings of songs with similar V-A coordinates,
    pushes apart songs with dissimilar coordinates.

    Args:
        threshold (float): Max Euclidean distance in V-A space to be considered
                           a 'positive pair'. Default 0.30.
        temperature (float): Softmax temperature for contrastive scaling.
    """
    def __init__(self, threshold: float = 0.30, temperature: float = 0.07):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings : (B, D) L2-normalized latent vectors
            labels     : (B, 2) valence-arousal scores in [0, 1]
        Returns:
            scalar loss
        """
        embeddings = F.normalize(embeddings, dim=1)
        B = embeddings.size(0)

        # Pairwise V-A distance → positive mask
        diff = labels.unsqueeze(0) - labels.unsqueeze(1)          # (B, B, 2)
        va_dist = torch.norm(diff, dim=2)                          # (B, B)
        pos_mask = (va_dist < self.threshold).float()
        pos_mask.fill_diagonal_(0)

        # Cosine similarity matrix
        sim = torch.mm(embeddings, embeddings.T) / self.temperature  # (B, B)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()                               # numerical stability

        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        n_pos = pos_mask.sum(dim=1)
        valid = n_pos > 0

        if valid.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        loss = -(pos_mask * log_prob).sum(dim=1)[valid] / n_pos[valid]
        return loss.mean()


# =============================================================================
# 2. NEW: Concordance Correlation Coefficient Loss
# =============================================================================

class CCCLoss(nn.Module):
    """
    Concordance Correlation Coefficient (CCC) Loss.

    CCC = (2 * cov(pred, target)) / (var(pred) + var(target) + (mean_pred - mean_target)^2)

    Unlike MSE, CCC simultaneously penalizes:
      - Low correlation
      - Mean bias (systematic shift)
      - Variance mismatch

    This is the standard loss in AVEC affective computing challenges and is
    especially beneficial for Valence prediction where systematic bias is common.

    Loss = 1 - mean(CCC_per_output)
    Range: [0, 2], with 0 meaning perfect agreement.
    """
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def _ccc_single(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute CCC for a single output dimension. Both tensors shape: (B,)"""
        pred_mean   = pred.mean()
        target_mean = target.mean()
        pred_var    = pred.var(unbiased=False)
        target_var  = target.var(unbiased=False)
        cov         = ((pred - pred_mean) * (target - target_mean)).mean()

        ccc = (2 * cov) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + self.eps)
        return ccc

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : (B, 2)  [arousal, valence] predictions
            target : (B, 2)  [arousal, valence] ground truth
        Returns:
            scalar loss = 1 - mean_CCC  (lower = better)
        """
        ccc_arousal = self._ccc_single(pred[:, 0], target[:, 0])
        ccc_valence = self._ccc_single(pred[:, 1], target[:, 1])
        mean_ccc    = (ccc_arousal + ccc_valence) / 2.0
        return 1.0 - mean_ccc

    def compute_ccc_scores(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Utility: returns CCC per dimension as a dict (for logging/reporting).
        Call this during evaluation, not training.
        """
        with torch.no_grad():
            return {
                "CCC_Arousal": self._ccc_single(pred[:, 0], target[:, 0]).item(),
                "CCC_Valence": self._ccc_single(pred[:, 1], target[:, 1]).item(),
            }


# =============================================================================
# 3. NEW: Differentiable Rank Loss (Spearman-proxy)
# =============================================================================

class RankLoss(nn.Module):
    """
    Differentiable Ranking Loss — Spearman Correlation Proxy.

    Preserves ordinal structure of predictions: even when absolute regression
    is noisy, this ensures the model correctly ranks songs from low→high emotion.
    This is critical for Phase C retrieval quality.

    Uses a soft-rank approximation via pairwise comparison sigmoid.

    Loss = 1 - mean(SoftSpearman_per_output)
    """
    def __init__(self, temperature: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def _soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable soft rank via pairwise sigmoid comparisons.
        x: (B,) → soft_ranks: (B,)
        """
        # pairwise differences: (B, B)
        diff = x.unsqueeze(0) - x.unsqueeze(1)
        # soft indicator: P(x_i > x_j)
        soft_gt = torch.sigmoid(diff / self.temperature)
        # soft rank = sum of P(x_i > x_j) for all j
        soft_ranks = soft_gt.sum(dim=1)
        return soft_ranks

    def _soft_spearman(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Soft Spearman for a single output. Both (B,)"""
        rank_pred   = self._soft_rank(pred)
        rank_target = self._soft_rank(target)

        rank_pred_mean   = rank_pred.mean()
        rank_target_mean = rank_target.mean()

        cov = ((rank_pred - rank_pred_mean) * (rank_target - rank_target_mean)).mean()
        std_pred   = (rank_pred.var(unbiased=False)   + self.eps).sqrt()
        std_target = (rank_target.var(unbiased=False) + self.eps).sqrt()

        return cov / (std_pred * std_target)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred   : (B, 2)
            target : (B, 2)
        Returns:
            scalar loss = 1 - mean_SoftSpearman
        """
        sp_arousal = self._soft_spearman(pred[:, 0], target[:, 0])
        sp_valence = self._soft_spearman(pred[:, 1], target[:, 1])
        mean_sp    = (sp_arousal + sp_valence) / 2.0
        return 1.0 - mean_sp


# =============================================================================
# 4. NEW: HybridLoss — unified drop-in loss for mainB.py
# =============================================================================

class HybridLoss(nn.Module):
    """
    Unified loss combining MSE + CCC + RankLoss + SupCR.

    Total = w_mse * MSE
          + w_ccc * CCC
          + w_rank * RankLoss
          + w_supcr * SupCRLoss   (only when use_supcr=True)

    Default weights are tuned for PMEmo-scale datasets (~800 samples).
    You can override via constructor args.

    Usage in training loop:
        criterion = HybridLoss(use_supcr=True)
        loss, components = criterion(preds, latent, targets)
        loss.backward()
    """
    def __init__(
        self,
        w_mse:   float = 1.0,
        w_ccc:   float = 0.5,
        w_rank:  float = 0.3,
        w_supcr: float = 0.1,
        use_supcr: bool = True,
        supcr_threshold: float = 0.30,
    ):
        super().__init__()
        self.w_mse   = w_mse
        self.w_ccc   = w_ccc
        self.w_rank  = w_rank
        self.w_supcr = w_supcr
        self.use_supcr = use_supcr

        self.mse_loss   = nn.MSELoss()
        self.ccc_loss   = CCCLoss()
        self.rank_loss  = RankLoss()
        self.supcr_loss = SupCRLoss(threshold=supcr_threshold)

    def forward(
        self,
        pred:      torch.Tensor,   # (B, 2)
        latent:    torch.Tensor,   # (B, D)
        target:    torch.Tensor,   # (B, 2)
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss : scalar, call .backward() on this
            components : dict of individual loss values (for logging)
        """
        l_mse  = self.mse_loss(pred, target)
        l_ccc  = self.ccc_loss(pred, target)
        l_rank = self.rank_loss(pred, target)

        total = self.w_mse * l_mse + self.w_ccc * l_ccc + self.w_rank * l_rank

        l_supcr = torch.tensor(0.0, device=pred.device)
        if self.use_supcr:
            l_supcr = self.supcr_loss(latent, target)
            total   = total + self.w_supcr * l_supcr

        components = {
            "loss_mse":   l_mse.item(),
            "loss_ccc":   l_ccc.item(),
            "loss_rank":  l_rank.item(),
            "loss_supcr": l_supcr.item(),
            "loss_total": total.item(),
        }
        return total, components
