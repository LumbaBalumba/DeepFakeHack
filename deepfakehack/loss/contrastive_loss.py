import torch
import torch.nn as nn

from oml.utils.misc_torch import elementwise_dist


class ContrastiveLoss(nn.Module):

    def __init__(self, alpha=1.0):
        super().__init__()
        self._alpha = alpha
        self._last_logs = {}

    def forward(self, positive: torch.tensor, negative: torch.tensor):
        assert positive.shape == negative.shape

        dists = elementwise_dist(positive, negative) * self._alpha

        dists = torch.relu(dists)

        self._last_logs = {"avg_dist": float(dists.mean().item())}

        return torch.mean(dists)

    @property
    def last_logs(self):
        return self._last_logs
