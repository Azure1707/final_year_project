import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedSoftVotingEnsemble(nn.Module):
    def __init__(self, models, weights=None):
        super().__init__()

        self.models = nn.ModuleList(models)

        if weights is None:
            weights = [1.0] * len(models)

        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum()

        self.register_buffer("weights", weights)

        for model in self.models:
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

    def forward(self, x):
        weighted_probs = torch.zeros(
            x.size(0),
            2,
            device=x.device
        )

        for model, weight in zip(self.models, self.weights):
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            weighted_probs += weight * probs

        return weighted_probs

