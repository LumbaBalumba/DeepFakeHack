from oml import datasets as d
from oml.samplers import BalanceSampler
import pandas as pd

from deepfakehack.loss.simple_sampler import SimpleSampler


class OMLDataset:
    def __init__(self, source_train, source_val, transform, sampler: str, params: dict):
        df_train, df_val = pd.read_csv(source_train), pd.read_csv(source_val)
        self.train = d.ImageLabeledDataset(df_train, transform=transform)
        self.val = d.ImageQueryGalleryLabeledDataset(df_val, transform=transform)

        if sampler == "SimpleSampler":
            self.sampler = SimpleSampler(self.train.get_labels(), **params)
        else:
            self.sampler = BalanceSampler(self.train.get_labels(), **params)
