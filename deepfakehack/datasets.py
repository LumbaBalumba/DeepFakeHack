from dataclasses import dataclass
import pandas as pd

from oml.samplers import BalanceSampler
from oml import datasets as d
from deepfakehack.loss.simple_sampler import SimpleSampler


@dataclass
class OMLDataset:
    def __init__(self, transform, sampler: str, params: dict):
        df_train, df_val = pd.read_csv(
            "./data/datasplit/train_with_meta.csv"
        ), pd.read_csv("./data/datasplit/val_with_meta.csv")
        self.train = d.ImageLabeledDataset(df_train, transform=transform)
        self.val = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

        if sampler == "SimpleSampler":
            self.sampler = SimpleSampler(self.train.get_labels(), **params)
        else:
            self.sampler = BalanceSampler(self.train.get_labels(), **params)
