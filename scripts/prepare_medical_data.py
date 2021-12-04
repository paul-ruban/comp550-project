import argparse
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from src.utils.json_utils import write_to_json

SMOKERS = {
    "unknown": 0,
    "non-smoker": 1,
    "past-smoker": 2,
    "smoker": 3,
    "current-smoker": 4,
}

DATA_TYPE = {"training", "test"}


def parse_args():
    parser = argparse.ArgumentParser(
        "Script for cleaning and preparing the medical.xml files into training and test."
    )
    parser.add_argument(
        "-i",
        "--input_data_path",
        type=str,
        help="Input data path. Must be an xml file.",
    )
    parser.add_argument(
        "-t", "--data_type", type=str, help="Dataset type either training or test."
    )
    parser.add_argument(
        "-o",
        "--output_folder_path",
        type=str,
        help="Path to the output folder that will contain the resulting datasets.",
    )
    parser.add_argument(
        "-n",
        "--number_of_folds",
        type=int,
        help="Number of folds to split the training. If n = 1 do a normal train val split.",
        required=False,
    )
    args = parser.parse_args()
    input_data_path, data_type, output_folder_path, number_of_folds = (
        args.input_data_path,
        args.data_type,
        args.output_folder_path,
        args.number_of_folds,
    )
    assert os.path.exists(input_data_path) and Path(input_data_path).suffix == ".xml"
    assert data_type in DATA_TYPE
    assert os.path.exists(output_folder_path)
    assert (data_type == "test" and number_of_folds is None) or (
        data_type == "training" and number_of_folds >= 1
    )
    return input_data_path, data_type, output_folder_path, number_of_folds


def clean_xml_file(input_data_path):
    # Get input data as xml tree
    tree = ET.parse(input_data_path)
    # First retrieve the y values
    y = []
    for root in tree.getroot():
        for descendant in root.iter():
            if descendant.tag.lower() == "smoking":
                y.append(descendant.attrib["STATUS"].lower().replace(" ", "-"))
    # Then retrieve the text, replacing newlines as spaces
    X = []
    for root in tree.getroot():
        for descendant in root.iter():
            if descendant.tag.lower() == "text":
                X.append(descendant.text.replace("\n", " ").lower())
    # Remove the header
    for i, x in enumerate(X):
        match = re.search(
            r"((discharge date)[^a-zA-Z]*|registration date[^a-zA-Z]*(AM|PM))",
            x,
            flags=re.IGNORECASE,
        )
        if match is None:
            print(f"You will need to handle {i} manually")
        else:
            end_index = match.end()
            X[i] = x[end_index:].strip()
    # Remove the footer and remove the **** strings
    X = [x.replace("[ report_end ]", "").replace("*", "") for x in X]
    # Randomly shuffle and return as numpy arrays, because everyone loves numpy ;)
    X, y = np.array(X), np.array(y)
    assert len(X.shape) == len(y.shape) == 1
    assert X.size == y.size
    index_permutation = np.random.permutation(len(X))
    X, y = X[index_permutation], y[index_permutation]
    return X, y


def write_train_to_output(X, y, number_of_folds, output_folder_path, random_seed=42):
    data_index_array = np.arange(len(X))
    # Do k-fold if k > 1
    if number_of_folds == 1:
        # Hard code the train-val split as 90-10
        train_idx, val_idx = train_test_split(
            data_index_array, train_size=0.9, test_size=0.1, random_state=random_seed
        )
        X_train, y_train, X_val, y_val = (
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
        )
        # Write the split to output files
        with open(os.path.join(output_folder_path, "training_text.txt"), "w") as f:
            for txt in X_train:
                f.write(f"{txt}\n")
        with open(os.path.join(output_folder_path, "training_labels.txt"), "w") as f:
            for label in y_train:
                f.write(f"{label}\n")
        # Write the validation
        write_to_json(X_val, y_val, output_folder_path, json_name="validation.json")
    else:
        kf = KFold(n_splits=number_of_folds)
        splits = kf.split(data_index_array)
        for k, (train_idx, val_idx) in enumerate(splits):
            k += 1
            X_train, y_train, X_val, y_val = (
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx],
            )
            # Write the split to output files
            with open(
                os.path.join(output_folder_path, f"training_fold_{k}_text.txt"), "w"
            ) as f:
                for txt in X_train:
                    f.write(f"{txt}\n")
            with open(
                os.path.join(output_folder_path, f"training_fold_{k}_labels.txt"), "w"
            ) as f:
                for label in y_train:
                    f.write(f"{label}\n")
            # Write the validation
            write_to_json(
                X_val, y_val, output_folder_path, json_name=f"validation_fold_{k}.json"
            )


def write_test_to_output(X, y, output_folder_path):
    write_to_json(X, y, output_folder_path, json_name="test.json")


def main():
    # Parse arguments
    input_data_path, data_type, output_folder_path, number_of_folds = parse_args()
    # Clean the input and retrieve text and labels
    X, y = clean_xml_file(input_data_path=input_data_path)
    if data_type == "training":
        write_train_to_output(
            X, y, number_of_folds=number_of_folds, output_folder_path=output_folder_path
        )
    elif data_type == "test":
        write_test_to_output(X, y, output_folder_path=output_folder_path)
    else:
        raise ValueError("Unrecognized data type")


if __name__ == "__main__":
    main()
