import random
from pathlib import Path
import json
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.optim as optim_lib
import torch.optim.lr_scheduler as sched_lib
import numpy as np
from oml import miners
from oml import losses

from deepfakehack import models
from deepfakehack.loss.loss import DeepFakelossWithMiner
from deepfakehack.datasets import OMLDataset


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

    model = getattr(models, cfg_data["algorithm"]["name"])(
        cfg_data["model"], **cfg_data["algorithm"]["model_params"]
    ).to(cfg_data["learning_params"]["device"])

    if cfg_data["type"] == "oml":
        optimizer = getattr(optim_lib, cfg_data["opt_name"])(
            model.model.parameters(), **cfg_data["opt_params"]
        )
        scheduler = getattr(sched_lib, cfg_data["scheduler_name"])(
            optimizer, **cfg_data["scheduler_params"]
        )

        data = OMLDataset(
            model.transform, cfg_data["sampler_name"], cfg_data["loader_params"]
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
    elif cfg_data["type"] == "abstract":
        model.train_loop(...)


if __name__ == "__main__":
    main()
