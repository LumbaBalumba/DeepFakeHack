from itertools import combinations
from sys import maxsize
from typing import List, Tuple

import torch


class PositiveTripletMiner:
    def __init__(self, max_output_triplets: int = maxsize, device: str = "cpu"):
        self._max_out_triplets = max_output_triplets
        self._device = device

    def sample(
        self, features: torch.Tensor, labels: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors = []
        positives = []
        negatives = []

        positive_classes = set(label for label in labels if label % 2 == 0)

        for pos_class in positive_classes:
            pos_indices = (labels == pos_class).nonzero(as_tuple=True)[0]
            if len(pos_indices) < 2:
                continue

            for anchor_idx, positive_idx in combinations(pos_indices, 2):
                negative_classes = set(labels) - {pos_class}

                for neg_class in negative_classes:
                    neg_indices = (labels == neg_class).nonzero(as_tuple=True)[0]

                    for negative_idx in neg_indices:
                        anchors.append(features[anchor_idx])
                        positives.append(features[positive_idx])
                        negatives.append(features[negative_idx])

                        if len(anchors) >= self._max_out_triplets:
                            return (
                                torch.stack(anchors),
                                torch.stack(positives),
                                torch.stack(negatives),
                            )
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
