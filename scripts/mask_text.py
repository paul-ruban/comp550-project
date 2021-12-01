import json
import os
import pickle

from sklearn.model_selection import ParameterGrid
from src.data.data_loader import load_data
from src.models.masking import (
    FrequencyMask,
    LengthWindowMask,
    POSMask,
    RandomWindowMask,
    StopwordMask,
    WeightedFrequencyMask,
)


def main():
    # Load validation dataset
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    VALIDATION_DATA_PATH = os.path.join(cur_dir, "..", "data", "clean", "validation")
    X_val = load_data(VALIDATION_DATA_PATH)
    # Creating masking param grid
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
    # Create cache folders
    MASKING_PKL_DATA_PATH = os.path.join(cur_dir, "..", "src", "models", "pickle", "masking")
    if not os.path.exists(MASKING_PKL_DATA_PATH):
        os.mkdir(MASKING_PKL_DATA_PATH)
    LOG_DATA_PATH = os.path.join(cur_dir, "..", "logs", "masking")
    if not os.path.exists(LOG_DATA_PATH):
        os.mkdir(LOG_DATA_PATH)
    masking_dict_name = "masking_dict.pkl"
    compression_json_name = "compression_accuracies.json"
    # Create masking dataset, pickle it and log the accuracies to json
    MASK_TOKEN = "_"
    X_masked_dict = {}
    compression_score = []
    for masking in masking_grid_list:
        kwargs = {k: v for k, v in masking.items() if k != "mask_type"}
        mask = masking["mask_type"](mask_token=MASK_TOKEN, **kwargs)
        masking_key = (mask.__class__.__name__, *tuple(kwargs.items()))
        X_masked_dict[masking_key] = mask.mask(X_val)
        compression_score.append(
            {
                "mask_type": mask.__class__.__name__,
                **kwargs,
                "compression_accuracy": mask.compression_score(
                    X_val, X_masked_dict[masking_key]
                ),
            }
        )
    with open(os.path.join(MASKING_PKL_DATA_PATH, masking_dict_name), "wb") as f:
        pickle.dump(X_masked_dict, f)
    # Log the compression metrics
    with open(os.path.join(LOG_DATA_PATH, compression_json_name), "w") as f:
        for dict_ in compression_score:
            json.dump(dict_, f)
            f.write("\n")


if __name__ == "__main__":
    main()
