from ast import Tuple
from itertools import  product

from typing import List

import numpy as np
import torch

class DupletMiner():

    def __init__(self, max_output_triplets: int = float('inf'), device: str = "cpu"):
        self._max_out_triplets = max_output_triplets
        self._device = device

    def sample(self, features: torch.Tensor, labels: List[int]):
        labels = np.array(labels)
        
        # Разделяем фичи на positive и negative по чётности labels
        positive_mask = labels % 2 == 0
        negative_mask = labels % 2 != 0
        
        positive_features = features[positive_mask]
        negative_features = features[negative_mask]
        
        positive_labels = labels[positive_mask]
        negative_labels = labels[negative_mask]
        
        # Создаём списки для хранения пар
        positive_pairs = []
        negative_pairs = []
        
        # Перебираем все пары positive и negative классов
        for pos_label in np.unique(positive_labels):
            neg_label = pos_label + 1
            
            if neg_label not in negative_labels:
                continue
            
            # Получаем фичи для текущего positive и negative класса
            pos_class_features = positive_features[positive_labels == pos_label]
            neg_class_features = negative_features[negative_labels == neg_label]
            
            # Перебираем все попарные комбинации
            for pos_feat, neg_feat in product(pos_class_features, neg_class_features):
                positive_pairs.append(pos_feat)
                negative_pairs.append(neg_feat)
                
                # Ограничиваем количество пар, если превышен лимит
                if len(positive_pairs) >= self._max_out_triplets:
                    break
            if len(positive_pairs) >= self._max_out_triplets:
                break
        
        # Преобразуем списки в тензоры
        positive_pairs = torch.stack(positive_pairs).to(self._device)
        negative_pairs = torch.stack(negative_pairs).to(self._device)
        
        return positive_pairs, negative_pairs



