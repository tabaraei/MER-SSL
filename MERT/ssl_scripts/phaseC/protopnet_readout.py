"""
protopnet_readout.py — learnable-prototype quadrant profile for Phase C
========================================================================
Replaces the post-hoc 4-centroid `retriever.prototype_profile` (fixed K-means-style
centroids, 0.51 accuracy, lost to the majority baseline) with the trained Audio
ProtoPNet (0.728 raw / 0.545 balanced held-out, §4 §3).

For a query the readout returns, per Russell quadrant, the prototype-activation
(closeness to that quadrant's learned prototypes) and the predicted quadrant — an
ante-hoc, self-explaining classification (each prototype is, by construction,
evidence for one quadrant).

Operates on the query's MERT features (the ProtoPNet's own backbone), independent
of the retrieval encoder — it augments the explanation, it does not change k-NN.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_PHASEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phaseB")
if _PHASEB not in sys.path:
    sys.path.insert(0, _PHASEB)

from models_protopnet import AudioProtoPNet                         # noqa: E402

DEFAULT_CKPT = os.path.join(_PHASEB, "protopnet_final.pt")
QUADRANTS = ["HVHA (Happy)", "HVLA (Calm)", "LVHA (Angry)", "LVLA (Sad)"]


class ProtoPNetReadout:
    def __init__(self, ckpt_path=DEFAULT_CKPT, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"ProtoPNet deployment checkpoint not found: {ckpt_path}\n"
                f"  Train it first:  cd ../phaseB && python train_deploy_models.py")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        ppc = ckpt.get("protos_per_class", 5)
        self.model = AudioProtoPNet(protos_per_class=ppc).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.ppc = ppc
        print(f"  ✅ ProtoPNetReadout | {os.path.basename(ckpt_path)} | "
              f"{self.model.n_proto} prototypes ({ppc}/quadrant)")

    @torch.no_grad()
    def profile(self, x_mert):
        """x_mert: (25,1024) → dict with predicted quadrant + per-quadrant activation."""
        xm = (x_mert if torch.is_tensor(x_mert)
              else torch.as_tensor(x_mert)).unsqueeze(0).float().to(self.device)
        logits, dist, _ = self.model(xm)                 # dist: (1, n_proto) squared L2
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        dist = dist.squeeze(0).cpu().numpy()
        # per-quadrant activation = closeness to that quadrant's nearest prototype
        activation = {}
        for c in range(4):
            block = dist[c * self.ppc:(c + 1) * self.ppc]
            activation[QUADRANTS[c]] = float(np.exp(-block.min()))  # nearest prototype
        pred = int(np.argmax(logits.squeeze(0).cpu().numpy()))
        return {
            "predicted_quadrant": QUADRANTS[pred],
            "class_probabilities": {QUADRANTS[c]: float(probs[c]) for c in range(4)},
            "prototype_activation": activation,
            "method": "Audio ProtoPNet (learnable prototypes, L2-distance, ante-hoc)",
        }
