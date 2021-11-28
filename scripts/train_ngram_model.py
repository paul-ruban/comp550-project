import json
import os
from pathlib import Path
from typing import List
import dill as pickle

from nltk.lm import MLE, KneserNeyInterpolated, Lidstone
from sklearn.model_selection import ParameterGrid
from src.models.masking import (
    FrequencyMask,
    LengthWindowMask,
    POSMask,
    RandomWindowMask,
    StopwordMask,
    WeightedFrequencyMask,
)
from src.models.ngram_model import NGramModel
from src.utils.eval_metrics import harmonic_mean


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


def main():
    # Create grid of parameters to try
    masking_grid = [
        {
            "mask_type": [RandomWindowMask],
            "window_size": [10],
            "prop_masked": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        {
            "mask_type": [LengthWindowMask],
            "window_size": [10],
            "prop_masked": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        {
            "mask_type": [FrequencyMask],
            "prop_masked": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        {
            "mask_type": [WeightedFrequencyMask],
            "prop_masked": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        {
            "mask_type": [StopwordMask],
            "prop_masked": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        {
            "mask_type": [POSMask],
            "pos_list": [
                ("CC", "DT", "EX", "IN", "MD", "PDT", "PRP", "RP"),
                ("RB", "RBS", "RBR", "JJ", "JJR", "JJS"),
                ("NN", "NNP", "NNS"),
                ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"),
            ],
        },
    ]
    masking_grid_list = list(ParameterGrid(masking_grid))
    lm_grid = [
        {"lm": [MLE], "n": [1, 2, 3, 4, 5]},
        {"lm": [Lidstone], "n": [1, 2, 3, 4, 5], "gamma": [0.1, 0.5, 0.7, 1, 2, 3]},
        {
            "lm": [KneserNeyInterpolated],
            "n": [1, 2, 3, 4, 5],
            "discount": [0.001, 0.01, 0.1, 0.5, 1, 2, 3],
        },
    ]
    lm_grid_list = list(ParameterGrid(lm_grid))
    # Get train dataset
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    TRAINING_DATA_PATH = os.path.join(cur_dir, "..", "data", "clean", "training")
    X_train = load_data(TRAINING_DATA_PATH)
    # Load validation dataset
    VALIDATION_DATA_PATH = os.path.join(cur_dir, "..", "data", "clean", "validation")
    X_val = load_data(VALIDATION_DATA_PATH)
    # Create masking dataset
    MASKING_PKL_DATA_PATH = os.path.join(
        cur_dir, "..", "src", "models", "pickle", "ngram", "ngram_lm_masked_dict.pkl"
    )
    LOG_DATA_PATH = os.path.join(cur_dir, "..", "logs", "ngram")
    compression_json_name = "compression_accuracies.json"
    if os.path.exists(MASKING_PKL_DATA_PATH):
        with open(MASKING_PKL_DATA_PATH, "rb") as f:
            X_masked_dict = pickle.load(f)
        with open(os.path.join(LOG_DATA_PATH, compression_json_name), "r") as f:
            lines = f.readlines()
            compression_score = []
            for dict_ in lines:
                compression_score.append(json.loads(dict_))
    else:
        X_masked_dict = {}
        compression_score = []
        mask_token = "_"
        for masking in masking_grid_list:
            kwargs = {k: v for k, v in masking.items() if k != "mask_type"}
            mask = masking["mask_type"](mask_token=mask_token, **kwargs)
            masking_key = (mask.__class__.__name__, *tuple(kwargs.items()))
            X_masked_dict[masking_key] = mask.mask(X_val)
            compression_score.append(
                {
                    "mask_type": mask.__class__.__name__,
                    **kwargs,
                    "compression_accuracy": mask.compression_score(X_val, X_masked_dict[masking_key]),
                }
            )
        with open(MASKING_PKL_DATA_PATH, "wb") as f:
            pickle.dump(X_masked_dict, f)
        # Log the metrics
        with open(os.path.join(LOG_DATA_PATH, compression_json_name), "w") as f:
            for dict_ in compression_score:
                json.dump(dict_, f)
                f.write("\n")
    # Train the language models
    evaluation_dict = {}
    for i, params in enumerate(lm_grid_list):
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
        # Evaluate the model on each masking dataset
        for key, X_masked in X_masked_dict.items():
            # This appears to be a time bottleneck
            X_decoded = model.decode(X_masked)
            reconstruction_accuracy = model.accuracy_score(
                masking_token=mask_token,
                X_original=X_val,
                X_masked=X_masked,
                X_decoded=X_decoded,
            )
            similarity_score = model.similarity_score(
                masking_token=mask_token,
                X_original=X_val,
                X_masked=X_masked,
                X_decoded=X_decoded,
            )
            evaluation_dict[(f"lm_{i}", *key)] = {
                "reconstruction_accuracy": reconstruction_accuracy,
                "similarity_score": similarity_score,
                "compression_score": compression_score[key],
                "harmonic_accuracy_compression": harmonic_mean(
                    [reconstruction_accuracy, compression_score[key]]
                ),
                "harmonic_similarity_compression": harmonic_mean(
                    [similarity_score, compression_score[key]]
                ),
            }
            print(evaluation_dict)


if __name__ == "__main__":
    main()
