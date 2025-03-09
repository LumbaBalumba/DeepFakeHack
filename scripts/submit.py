import os
from typing import List
import json
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime

import torch
from torch.nn import functional as F
import pandas as pd
from oml import datasets as d
from oml.inference import inference
from oml.registry import get_transforms_for_pretrained

import models


def parse_args():
    parser = ArgumentParser(description="Training pipeline")
    parser.add_argument("--cfg-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    return args


def create_sample_sub(pair_ids: List[str], sim_scores: List[float]):
    sub_sim_column = "similarity"
    id_column = "pair_id"
    return pd.DataFrame({id_column: pair_ids, sub_sim_column: sim_scores})


def main() -> None:
    if not os.path.exists("result"):
        os.makedirs("result")

    args = parse_args()
    with open(f"configs/{args.cfg_path}", encoding="utf8") as f:
        cfg_data = json.load(f)

    device = cfg_data["learning_params"]["device"]
    test_path = "./data/datasplit/test.csv"

    model = getattr(models, cfg_data["algorithm"]["name"])(
        cfg_data["model"], **cfg_data["algorithm"]["model_params"]
    ).model
    state_dict = torch.load(f"./model_weights/{args.model_path}", map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    transform, _ = get_transforms_for_pretrained("resnet18_imagenet1k_v1")

    df_test = pd.read_csv(test_path)
    test = d.ImageQueryGalleryLabeledDataset(df_test, transform=transform)
    embeddings = inference(model, test, batch_size=32, num_workers=0, verbose=True)
    embeddings = F.normalize(embeddings)

    e1 = embeddings[::2]
    e2 = embeddings[1::2]
    sim_scores = F.cosine_similarity(e1, e2).detach().cpu().numpy()

    pair_ids = df_test["label"].apply(lambda x: f"{x:08d}").to_list()
    pair_ids = pair_ids[::2]

    sub_df = create_sample_sub(pair_ids, sim_scores)
    try:
        el = str(args.model_path)[str(args.model_path).rfind("/") : -4]
    except IndexError:
        el = "__"
    sub_df.to_csv(
        f"./result/{cfg_data['model']}_ep_{el}_"
        + str(datetime.now())
        .split(".", maxsplit=1)[0]
        .replace(" ", "_")
        .replace("-", "")
        .replace(":", "_")
        + ".csv",
        index=False,
    )


if __name__ == "__main__":
    main()
