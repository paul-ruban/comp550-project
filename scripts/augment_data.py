import argparse
import os
from typing import Union, List
import datetime
from tqdm import tqdm

import numpy as np
from sklearn.model_selection import ParameterGrid
from src.augmentation.augmentation import Augmentation

cur_dir = os.path.dirname(os.path.abspath(__file__))

DATA_TYPES = {"polarity", "articles", "smokers"}


PATH_DICTS = {
    "polarity": {
        "input_training_text_path": os.path.join(
            cur_dir,
            "..",
            "data",
            "rt-polaritydata",
            "augmentation",
            "training_text.txt",
        ),
        "input_training_label_path": os.path.join(
            cur_dir,
            "..",
            "data",
            "rt-polaritydata",
            "augmentation",
            "training_labels.txt",
        ),
        # TODO: Modify this to the mila cluster
        "output_folder_path": "/home/mila/c/cesare.spinoso/scratch/datasets_550/polarity",
    },
    "articles": {
        "input_training_text_path": os.path.join(
            cur_dir, "..", "data", "articles", "augmentation", "training_text.txt"
        ),
        "input_training_label_path": os.path.join(
            cur_dir,
            "..",
            "data",
            "articles",
            "augmentation",
            "training_labels.txt",
        ),
        # TODO: Modify this to the mila cluster
        "output_folder_path": "/home/mila/c/cesare.spinoso/scratch/datasets_550/articles",
    },
    "smokers": {
        "input_training_text_path": [
            os.path.join(
                cur_dir,
                "..",
                "data",
                "smokers",
                "augmentation",
                f"training_fold_{k}_text.txt",
            )
            for k in range(1, 6)
        ],
        "input_training_label_path": [
            os.path.join(
                cur_dir,
                "..",
                "data",
                "smokers",
                "augmentation",
                f"training_fold_{k}_labels.txt",
            )
            for k in range(1, 6)
        ],
        # TODO: Modify this to the mila cluster
        "output_folder_path": "/home/mila/c/cesare.spinoso/scratch/datasets_550/smokers",
    },
}

# Path to logger folder
PATH_TO_LOGGER_FOLDER = os.path.join(cur_dir, "..", "logs", "augmentation")

# TODO: Work on augmentation grid
AUGMENTATION_GRID = [
    {
        "augmentation_type": ["random_swap"],
        "num_samples": [1, 3, 5],
        "aug_p": [0.1, 0.25, 0.5, 0.75, 0.9],
    },
    {
        "augmentation_type": ["random_delete"],
        "num_samples": [1, 3, 5],
        "aug_p": [0.1, 0.25, 0.5, 0.75, 0.9],
    },
    {
        "augmentation_type": ["synonym_wordnet"],
        "num_samples": [1, 3, 5],
        "aug_p": [0.1, 0.25, 0.5, 0.75, 0.9],
        "stopwords_regex": [r".*[^a-zA-Z].*"] # skip non-alpha words
    },
    {
        "augmentation_type": ["synonym_word2vec"],
        "num_samples": [1, 3, 5],
        "aug_p": [0.1, 0.25, 0.5, 0.75, 0.9],
        "top_k": [10, 100, None]
    },
    {
        "augmentation_type": ["backtranslation"],
        "num_samples": [1, 3, 5],
    },
]


def parse_args():
    parser = argparse.ArgumentParser("Aguments the dataset of the passed data type")
    parser.add_argument(
        "-t", "--data_type", help=f"Data type must be one of {DATA_TYPES}"
    )
    args = parser.parse_args()
    data_type = args.data_type
    assert data_type in DATA_TYPES
    return data_type


def load_data_set(
    path_to_text: Union[str, list], path_to_label: Union[str, list]
) -> List[np.array]:
    X = (
        [np.loadtxt(path_to_text, dtype="object", delimiter="\n")]
        if isinstance(path_to_label, str)
        else [np.loadtxt(path, dtype="object", delimiter="\n") for path in path_to_text]
    )
    y = (
        [np.loadtxt(path_to_label, dtype="object", delimiter="\n")]
        if isinstance(path_to_label, str)
        else [
            np.loadtxt(path, dtype="object", delimiter="\n") for path in path_to_label
        ]
    )
    return X, y


def main():
    # Get data type
    data_type = parse_args()
    # Create augmentation grid
    grid_list = list(ParameterGrid(AUGMENTATION_GRID))
    # Fetch data
    X_list, y_list = load_data_set(
        path_to_text=PATH_DICTS[data_type]["input_training_text_path"],
        path_to_label=PATH_DICTS[data_type]["input_training_label_path"],
    )
    # Augment data
    # Use a time signature for the logger
    time_now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    for aug_id, aug_kwargs in tqdm(enumerate(grid_list)):
        print(f"Augmentation {data_type} with parameters {aug_kwargs}")
        aug = Augmentation(
            augmentation_type=aug_kwargs["augmentation_type"],
            num_samples=aug_kwargs["num_samples"],
        )
        other_kwargs = {
            k: v
            for k, v in aug_kwargs.items()
            if k != "augmentation_type" and k != "num_samples"
        }
        for i, (X, y) in enumerate(zip(X_list, y_list)):
            X_aug, y_aug = aug.augment(X, y, **other_kwargs)
            aug.to_json(
                path_to_folder=os.path.join(
                    PATH_DICTS[data_type]["output_folder_path"],
                    f"augmentation_{aug_id}",
                ),
                X_initial=X,
                y_initial=y,
                X_aug=X_aug,
                y_aug=y_aug,
                name="train" if len(X_list) == 1 else f"train_fold_{i+1}",
            )
            aug.log_to_json(
                id=aug_id,
                time_now=time_now,
                path_to_json_log=os.path.join(
                    PATH_TO_LOGGER_FOLDER, f"{data_type}.json"
                ),
            )

if __name__ == "__main__":
    main()
