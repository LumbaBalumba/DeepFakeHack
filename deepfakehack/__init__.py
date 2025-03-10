import torch.optim as optim_lib
import torch.optim.lr_scheduler as sched_lib

from deepfakehack import models


class ModelBuilder:
    @staticmethod
    def build(cfg_data):
        return getattr(models, cfg_data["algorithm"]["name"])(
            cfg_data["model"], **cfg_data["algorithm"]["model_params"]
        ).to(cfg_data["learning_params"]["device"])


class OptimizerBuilder:
    @staticmethod
    def build(cfg_data, model):
        return getattr(optim_lib, cfg_data["opt_name"])(
            model.parameters(), **cfg_data["opt_params"]
        )


class SchedulerBuilder:
    @staticmethod
    def build(cfg_data, optimizer):
        return getattr(sched_lib, cfg_data["scheduler_name"])(
            optimizer, **cfg_data["scheduler_params"]
        )
