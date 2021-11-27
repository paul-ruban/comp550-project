import os
import random
from pathlib import Path

from src.models.ngram_model import NgramModel
from nltk.lm import MLE, Lidstone, KneserNeyInterpolated
from sklearn.model_selection import ParameterGrid
from typing import List


def load_data(data_path: str) -> List[str]:
    """
    Loads data from the given path.
    """
    data_path = Path(data_path)
    file_paths = data_path.glob("**/*.txt")
    data = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            data.append(f.read())
    return data


def mask_data(data: List[str], masking_strategy: str, **kwargs) -> List[str]:
    """
    Masks the given data according to the given masking strategy.
    """
    if masking_strategy == "random":
        return mask_random(data, **kwargs)
    elif masking_strategy == "longest_word":
        return mask_longest_word(data, **kwargs)
    elif masking_strategy == "most_frequent_word":
        return mask_most_frequent_word(data, **kwargs)
    elif masking_strategy == "modified_tfidf":
        return mask_modified_tfidf(data, **kwargs)
    else:
        raise ValueError(f"Invalid masking strategy: {masking_strategy}")


def mask_random(
    data: List[str], window_size: int = 10, num_of_masks: int = 1, prop_masked: float = None
) -> List[str]:
    assert (num_of_masks is not None and prop_masked is None) or (
        num_of_masks is None and prop_masked is not None
    )
    masked_data = []
    random.seed(42)
    for elt in data:
        num_of_masks = int(len(elt) * prop_masked) if prop_masked is not None else num_of_masks
        pass




def mask_longest_word(data: List[str]) -> List[str]:
    pass


def mask_most_frequent_word(data: List[str]) -> List[str]:
    pass


def mask_modified_tfidf(data: List[str]) -> List[str]:
    pass


def main():
    # Create grid of parameters to try
    grid = [
        {"lm": MLE, "n": [1, 2, 3, 4, 5]},
        {"lm": Lidstone, "n": [1, 2, 3, 4, 5], "gamma": [0.1, 0.3, 0.5, 0.7, 1, 2, 3]},
        {
            "lm": KneserNeyInterpolated,
            "n": [1, 2, 3, 4, 5],
            "discount": [0.001, 0.01, 0.1, 0.5, 1, 2, 3],
        },
    ]
    grid_list = list(ParameterGrid(grid))
    # Get train dataset
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    TRAINING_DATA_PATH = os.path.join(cur_dir, "..", "data", "clean", "training")
    X_train = load_data(TRAINING_DATA_PATH)
    # Create validation datasets (i.e. with masking strategies)
    VALIDATION_DATA_PATH = os.path.join(cur_dir, "..", "data", "clean", "validation")
    data_val = load_data(VALIDATION_DATA_PATH)
    masking_strategies = ["random", "longest_word", "most_frequent_word", "modified_tfidf"]
    X_val = {}
    for masking_strategy in masking_strategies:
        X_val[masking_strategy] = mask_data(data_val, masking_strategy)
    # Train the language models
    for params in grid_list:
        kwargs = {k: v for k, v in params.items() if k != "lm" and k != "n"}
        model = NgramModel(n=params["n"])
        lm = model.fit(X_train, lm=params["lm"], **kwargs)
        print(lm)


if __name__ == "__main__":
    main()
