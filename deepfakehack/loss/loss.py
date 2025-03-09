from typing import Tuple

import torch
import torch.nn as nn

from oml.losses.triplet import TripletLoss
from deepfakehack.loss.contrastive_loss import ContrastiveLoss

from .triplet_miner import PositiveTripletMiner
from .duplet_miner import DupletMiner

class DeepFakeloss(nn.Module):
    def __init__(self, 
                 gamma = 1.0,
                 margin = 0.1
                 ):
        super().__init__()
        self.tri_loss = TripletLoss(margin, need_logs = True)
        self.conrt_loss = ContrastiveLoss()
        self._gamma = gamma
        self._last_logs = {}

    def forward(self, triplet : Tuple[torch.tensor, torch.tensor, torch.tensor], pair : Tuple[torch.tensor, torch.tensor]):
        anchor, positive, negative = triplet
        trues, fakes = pair

        loss = self.tri_loss(anchor, positive, negative) - self.conrt_loss(trues, fakes)
        self._last_logs = self.tri_loss.last_logs | self.conrt_loss.last_logs
        return loss
    
    @property
    def last_logs(self):
        return self._last_logs
    
class DeepFakelossWithMiner(nn.Module):
    def __init__(self, 
                 gamma = 1.0
                 ):
        super().__init__()
        self.loss= DeepFakeloss(gamma)
        self._gamma = gamma
        self.tri_miner = PositiveTripletMiner()
        self.dup_miner = DupletMiner()
        self._last_logs = {}

    def forward(self, features, labels):
        triplet = self.tri_miner.sample(features, labels)
        duplet = self.dup_miner.sample(features, labels)

        loss = self.loss(triplet, duplet)
        self._last_logs = self.loss.last_logs
        return loss
    
    @property
    def last_logs(self):
        return self._last_logs