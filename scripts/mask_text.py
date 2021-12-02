import json
import os
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path

from sklearn.model_selection import ParameterGrid
from src.data.data_loader import load_data
from src.models.masking import (
    FrequencyMask,
    LengthWindowMask,
    Mask,
    POSMask,
    RandomWindowMask,
    StopwordMask,
    WeightedFrequencyMask,
)


def parse_args():
    # Set up default arguments
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # Input folder (that contains files to mask)
    VALIDATION_FOLDER_DATA_PATH = os.path.join(
        cur_dir, "..", "data", "clean", "validation"
    )
    # Pickler folder + file name that will contain the dictionary of masked data
    # Where the key is the type of masking and the value is the list of masked texts
    MASKING_PKL_FOLDER_DATA_PATH = os.path.join(
        cur_dir, "..", "src", "models", "pickle", "masking"
    )
    masking_dict_name = "masking_dict.pkl"
    masking_pkl_file_path = os.path.join(
        cur_dir, MASKING_PKL_FOLDER_DATA_PATH, masking_dict_name
    )
    # Log folder + file name will contain the compression accuracies = 1 - compression ratio
    LOG_DATA_FOLDER_PATH = os.path.join(cur_dir, "..", "logs", "masking")
    compression_json_name = "compression_accuracies.json"
    log_file_path = os.path.join(LOG_DATA_FOLDER_PATH, compression_json_name)
    # Set up the command line parser
    parser = argparse.ArgumentParser(description="Clean texts preprocessing script")
    parser.add_argument(
        "-i",
        "--input_path",
        required=False,
        help="Path of text FOLDERS to mask.",
        default=VALIDATION_FOLDER_DATA_PATH,
    )
    parser.add_argument(
        "-p",
        "--pickle_file_path",
        required=False,
        help="Output path to the pickle file (pkl).",
        default=masking_pkl_file_path,
    )
    parser.add_argument(
        "-l",
        "--log_file_path",
        required=False,
        help="Output path to the log file (json).",
        default=log_file_path,
    )
    # Parse arguments
    args = parser.parse_args()
    input_path = Path(args.input_path)
    pickle_file_path = Path(args.pickle_file_path)
    log_file_path = Path(args.log_file_path)

    # Do some checks on path
    assert (
        input_path.exists() and input_path.is_dir()
    ), "Input path does not exist or is not a directory."
    if not os.path.exists(pickle_file_path.parent):
        os.makedirs(pickle_file_path.parent)
    if not os.path.exists(log_file_path.parent):
        os.makedirs(log_file_path.parent)
    return input_path, pickle_file_path, log_file_path


def main():
    # Get path arguments
    input_path, pickle_file_path, log_file_path = parse_args()
    # Load test to mask
    X_val = load_data(input_path)
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
    # Create masking dataset, pickle it and log the accuracies to json
    MASK_TOKEN = "_"
    X_masked_dict = {}
    compression_score = []
    # Baseline of no masking just gzip
    no_mask = Mask()
    compression_score.append(
        {
            "mask_type": "no_mask",
            "compression_accuracy": no_mask.compression_score(X_val, X_val)
        }
    )
    # Mask with each paradigm
    print("Masking texts :)")
    for masking in tqdm(masking_grid_list):
        kwargs = {k: v for k, v in masking.items() if k != "mask_type"}
        mask = masking["mask_type"](mask_token=MASK_TOKEN, use_pos_as_mask=True, **kwargs)
        masking_key = (mask.__class__.__name__, *tuple(kwargs.items()))
        X_masked_dict[masking_key] = mask.mask(X_val, X_val)
        compression_score.append(
            {
                "mask_type": mask.__class__.__name__,
                **kwargs,
                "compression_accuracy": mask.compression_score(
                    X_val, X_masked_dict[masking_key]
                ),
            }
        )
    with open(pickle_file_path, "wb") as f:
        pickle.dump(X_masked_dict, f)
    # Log the compression metrics
    with open(log_file_path, "w") as f:
        for dict_ in compression_score:
            json.dump(dict_, f)
            f.write("\n")


if __name__ == "__main__":
    main()
