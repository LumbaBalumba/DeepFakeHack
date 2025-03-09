import pandas as pd
import numpy as np
import json

path = "./data"

val_path = path + "datasplit/val.csv"
meta_path = path + "train/meta.json"

new_val_path = path + "datasplit/val_with_meta.csv"

with open(meta_path) as f:
    meta_data = json.load(f)

val_df = pd.read_csv(val_path)


def row_handler(row):
    template = "./data/train/images/"
    return 2 * meta_data[row["path"][len(template) :]] - 1


mask = np.array(val_df.apply(row_handler, axis=1))

labels = val_df["label"].to_numpy()
labels += 1


labels = labels * mask + np.max(labels)
val_df["label"] = labels

lbs, counts = np.unique(labels, return_counts=True)
lbs = lbs[counts < 2]
filtered_df = val_df[~val_df["label"].isin(lbs)]
filtered_df.to_csv(new_val_path, index=False, mode="w")
