import argparse
import os
import random
import json
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def load_files_in_directory(directory):
    return sorted(os.listdir(directory))


def generate_ssl_json(args):
    # args:
    #   args.path -> dataset root containing a "DENTAL" folder
    #   args.ratio -> fraction for validation split (e.g., 0.2)
    #   args.json -> output json path
    set_seed(42)

    dental_dir = args.path
    dental_files = load_files_in_directory(dental_dir)
    if not dental_files:
      raise ValueError(f"No images found in {dental_dir}")

    random.shuffle(dental_files)
    cut_index = int(args.ratio * len(dental_files))

    val_files = dental_files[:cut_index]
    train_files = dental_files[cut_index:]

    print(f"# Training samples: {len(train_files)}\t# Validation samples: {len(val_files)}")

    training = [{"image": [f"./DENTAL/{f}"]} for f in train_files]
    validation = [{"image": [f"./DENTAL/{f}"]} for f in val_files]

    data = {"training": training, "validation": validation}

    with open(args.json, "w") as f:
        json.dump(data, f, indent=2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON for a dataset")
    parser.add_argument(
        "--path", default="dataset/dataset0", type=str, help="Path to the images"
    )
    parser.add_argument(
        "--json", default="jsons/dataset0.json", type=str, help="Path to the JSON output"
    )
    parser.add_argument(
        "--ratio", default=0.1, type=float, help="Ratio of validation data"
    )
    parser.add_argument(
        "--folds", default=1, type=int, help="Number of folds"
    )
    args = parser.parse_args()

    args.path = "../DENTAL"
    args.json = "jsons"
    args.folds = 1

    generate_ssl_json(args)
