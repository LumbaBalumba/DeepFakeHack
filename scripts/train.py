from pathlib import Path
import json
from argparse import ArgumentParser
from datetime import datetime
import random

import numpy as np
import torch

from oml import miners
from oml import losses

from deepfakehack import ModelBuilder, OptimizerBuilder, SchedulerBuilder
from deepfakehack.datasets import OMLDataset
from deepfakehack.loss.loss import DeepFakelossWithMiner


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = ArgumentParser(description="Training pipeline")
    parser.add_argument("--cfg-path", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    with open(f"configs/{args.cfg_path}", encoding="utf8") as f:
        cfg_data = json.load(f)

    fix_seed(cfg_data["seed"])

    model = ModelBuilder.build(cfg_data)

    optimizer = OptimizerBuilder.build(cfg_data, model)
    scheduler = SchedulerBuilder.build(cfg_data, optimizer)

    data = OMLDataset(
        "./data/datasplit/train_with_meta.csv",
        "./data/datasplit/val_with_meta.csv",
        model.transform,
        cfg_data["sampler_name"],
        cfg_data["loader_params"],
    )
    if cfg_data["loss_name"] == "DeepFakeLoss":
        criterion = DeepFakelossWithMiner()
    else:
        loss = getattr(losses, cfg_data["loss_name"])
        criterion = loss(
            **cfg_data["loss_params"],
            **(
                {"miner": getattr(miners, cfg_data["miner_name"])()}
                if hasattr(loss, "miner")
                else {}
            ),
        )

    path2weights = (
        cfg_data["path2weights"]
        + "/"
        + cfg_data["model"]
        + "___"
        + str(datetime.now())
        .split(".", maxsplit=1)[0]
        .replace(" ", "_")
        .replace("-", "")
        .replace(":", "_")
    )

    model.training_oml(
        **cfg_data["learning_params"],
        dataset=data,
        model=model.model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        path2weights=path2weights,
    )


if __name__ == "__main__":
    main()
