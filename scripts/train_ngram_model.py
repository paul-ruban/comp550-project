import os
import random
from pathlib import Path

from src.models.ngram_model import NGramModel
from nltk.lm import MLE, Lidstone, KneserNeyInterpolated
from sklearn.model_selection import ParameterGrid
from typing import List


def load_data(data_path: str) -> List[str]:
    """
    Loads data from the given path as list of strings. Each
    string is a text.
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
    data: List[str],
    masking_char: str = "_",
    window_size: int = 10,
    num_of_masks: int = 1,
    prop_masked: float = None,
    random_seed: int = 42,
) -> List[str]:
    """
    Data is a list of strings, each string is a text. The text has already been cleaned to
    be tokenized and separated by spaces.
    """
    assert (num_of_masks is not None and prop_masked is None) or (
        num_of_masks is None and prop_masked is not None
    )
    masked_data = []
    random.seed(random_seed)
    for text in data:
        tokenized_text = text.split()
        num_of_masks = int(window_size * prop_masked) if prop_masked is not None else num_of_masks
        masked_indices = [
            j
            for i in range(int(len(tokenized_text) / window_size))
            for j in random.sample(range(i * window_size, (i + 1) * window_size), num_of_masks)
        ]
        masked_text = [
            masking_char if i in masked_indices else token for i, token in enumerate(tokenized_text)
        ]
        masked_data.append(" ".join(masked_text))
    return masked_data


def mask_longest_word(
    data: List[str],
    masking_char: str = "_",
    window_size: int = 10,
    num_of_masks: int = 1,
    prop_masked: float = None,
) -> List[str]:
    assert (num_of_masks is not None and prop_masked is None) or (
        num_of_masks is None and prop_masked is not None
    )
    masked_data = []
    for text in data:
        tokenized_text_with_index = list(enumerate(text.split()))
        num_of_masks = int(window_size * prop_masked) if prop_masked is not None else num_of_masks
        masked_indices = [
            j
            for i in range(int(len(tokenized_text_with_index) / window_size))
            for (j, _) in sorted(
                tokenized_text_with_index[i * window_size : (i + 1) * window_size],
                key=lambda x: len(x[1]),
                reverse=True,
            )[:num_of_masks]
        ]
        masked_text = [
            masking_char if i in masked_indices else token for i, token in tokenized_text_with_index
        ]
        masked_data.append(" ".join(masked_text))
    return masked_data


def mask_most_frequent_word(data: List[str]) -> List[str]:
    pass


def mask_modified_tfidf(data: List[str]) -> List[str]:
    pass


def mask_stopwords(data: List[str]) -> List[str]:
    pass


def mask_pos_tags(data: List[str]) -> List[str]:
    pass


def main():
    # Create grid of parameters to try
    grid = [
        {"lm": [MLE], "n": [1, 2, 3, 4, 5]},
        {"lm": [Lidstone], "n": [1, 2, 3, 4, 5], "gamma": [0.1, 0.5, 0.7, 1, 2, 3]},
        {
            "lm": [KneserNeyInterpolated],
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
        # TODO: Compress here so already have the ratio and don't need to decompress while training

    # Train the language models
    for i, params in enumerate(grid_list):
        print(f"Training n-gram language model with id = {i} with params : \n{params}")
        pickle_path = os.path.join(cur_dir, "..", "src", "models", "pickle", "ngram", f"lm_{i}.pkl")
        model = NGramModel(n=params["n"])
        if os.path.exists(pickle_path):
            model.load_model(pickle_path)
        else:
            kwargs = {k: v for k, v in params.items() if k != "lm" and k != "n"}
            model.fit(X_train, lm=params["lm"], **kwargs)
            model.save_model(
                pickle_model_path=os.path.join(
                    cur_dir, "..", "src", "models", "pickle", "ngram", f"lm_{i}.pkl"
                )
            )


if __name__ == "__main__":
    main()
