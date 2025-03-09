from itertools import combinations
from sys import maxsize

import numpy as np
import torch
from typing import List, Tuple

class PositiveTripletMiner:
    def __init__(self, max_output_triplets: int = maxsize, device: str = "cpu"):
        self._max_out_triplets = max_output_triplets
        self._device = device

    def sample(self, features: torch.Tensor, labels: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchors = []
        positives = []
        negatives = []

        # Преобразуем labels в тензор

        # Уникальные позитивные классы (четные метки)
        positive_classes = set(label for label in labels if label % 2 == 0)

        # print(positive_classes)
        # print(labels)

        for pos_class in positive_classes:
            # Индексы элементов текущего позитивного класса
            # print((labels == pos_class))
            pos_indices = (labels == pos_class).nonzero(as_tuple=True)[0]
            if len(pos_indices) < 2:
                continue  # Пропускаем классы с менее чем 2 элементами

            # Все возможные пары внутри позитивного класса
            for anchor_idx, positive_idx in combinations(pos_indices, 2):
                # Все негативные классы (все классы, кроме текущего позитивного)
                # print(anchor_idx, positive_idx)
                negative_classes = set(labels) - {pos_class}

                for neg_class in negative_classes:
                    # Индексы элементов текущего негативного класса
                    neg_indices = (labels == neg_class).nonzero(as_tuple=True)[0]

                    for negative_idx in neg_indices:
                        anchors.append(features[anchor_idx])
                        positives.append(features[positive_idx])
                        negatives.append(features[negative_idx])

                        # Ограничение на максимальное количество троек
                        if len(anchors) >= self._max_out_triplets:
                            return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)
        # print(anchors)
        # print(positives)
        # print(negatives)
        return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)