import argparse
import os
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from src.utils.json_utils import write_json_lines

POLARITY_LABELS = {"neg": 0, "pos": 1}
ARTICLES_LABELS = {
    "business": 0,
    "entertainment": 1,
    "politics": 2,
    "sport": 3,
    "tech": 4,
}
SMOKERS = {
    "unknown": 0,
    "non-smoker": 1,
    "past-smoker": 2,
    "smoker": 3,
    "current-smoker": 4,
}
TYPES_DATA_DICT = {
    "polarity": POLARITY_LABELS,
    "articles": ARTICLES_LABELS,
    "smokers": SMOKERS,
}
JSON_KEYS = ["id", "label", "text"]


def parse_args():
    parser = argparse.ArgumentParser(
        "Script for preparing the datasets as train_text.txt, train_label.txt, validation.json, testing.json"
    )
    parser.add_argument(
        "-t",
        "--type_data",
        type=str,
        default="polarity",
        help=f"Type of data to prepare must be one of {list(TYPES_DATA_DICT.keys())}",
    )
    parser.add_argument(
        "-i",
        "--input_folder_path",
        type=str,
        help="The input folder that will contain the text files as class_k.txt",
    )
    parser.add_argument(
        "-o",
        "--output_folder_path",
        type=str,
        help="Where to write train_text.txt, train_label.txt, validation.json, testing.json",
    )
    args = parser.parse_args()

    type_data = args.type_data
    input_folder_path = args.input_folder_path
    output_folder_path = args.output_folder_path

    assert (
        type_data in TYPES_DATA_DICT.keys()
    ), f"The type must be one of {TYPES_DATA_DICT}"
    assert os.path.exists(
        input_folder_path
    ), f"Input folder {input_folder_path} doesn't exist."
    assert all(
        any(labels in file for labels in TYPES_DATA_DICT[type_data].keys())
        for file in os.listdir(input_folder_path)
    ), f"The input folder {input_folder_path} must only contain files {list(TYPES_DATA_DICT[type_data].keys())}.txt"
    assert os.path.exists(
        output_folder_path
    ), f"Output folder {output_folder_path} doesn't exist."

    return type_data, input_folder_path, output_folder_path


def extract_text_and_labels(type_data, input_folder_path):
    dict_data = dict.fromkeys(TYPES_DATA_DICT[type_data].keys(), None)
    # Import texts
    for file in os.listdir(input_folder_path):
        for label_class in dict_data.keys():
            if label_class in file:
                dict_data[label_class] = np.loadtxt(
                    os.path.join(input_folder_path, file),
                    dtype=str,
                    delimiter="\n",
                )
    # Compose the lists of texts
    text = np.hstack([text_array for text_array in dict_data.values()])
    # Compose the labels
    labels = np.hstack(
        [
            np.full(
                shape=text_array.size,
                fill_value=TYPES_DATA_DICT[type_data][label_class],
            )
            for label_class, text_array in dict_data.items()
        ]
    )
    assert (
        text.size == labels.size
    ), "Uh oh! The number of labels is different than the number of text files!"
    assert (
        len(text.shape) == len(labels.shape) == 1
    ), "labels and text should have 1 dimension"

    return text, labels


def get_train_test_index(
    input_size: int,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[list, list, list]:
    """Returns the indices to use for the training, validation and testing dataset.

    Args:
        text (int): Size of the input.
        train_size (float, optional): Size of the training set. Defaults to 0.8.
        val_size (float, optional): Size of the validation set. Defaults to 0.1.
        test_size (float, optional): Size of the test set. Defaults to 0.1.
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[list, list, list]: Lists of indices for the training, validation and test
        set.
    """
    num_list = [train_size, val_size, test_size]
    assert all(isinstance(num, float) for num in num_list)
    assert all(0 < num < 1 for num in num_list)
    assert sum(num_list) == 1
    assert isinstance(random_state, int)
    data_index = np.arange(input_size)
    train_idx, val_idx = train_test_split(
        data_index, train_size=train_size, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        val_idx, train_size=val_size / (1 - train_size), random_state=random_state
    )
    return train_idx, val_idx, test_idx


def write_to_output(
    text: np.array,
    labels: np.array,
    train_idx: np.array,
    val_idx: np.array,
    test_idx: np.array,
    output_folder_path: str,
):
    # Write validation and test json
    val_dict = [
        {"id": id, "label": int(label), "text": str(text)}
        for id, (label, text) in enumerate(zip(labels[val_idx], text[val_idx]))
    ]
    write_json_lines(val_dict, os.path.join(output_folder_path, "validation.json"))
    test_dict = [
        {"id": id, "label": int(label), "text": str(text)}
        for id, (label, text) in enumerate(zip(labels[test_idx], text[test_idx]))
    ]
    write_json_lines(test_dict, os.path.join(output_folder_path, "test.json"))
    # Write train text and labels
    with open(os.path.join(output_folder_path, "training_text.txt"), "w") as f:
        for txt in text[train_idx]:
            f.write(f"{txt}\n")
    with open(os.path.join(output_folder_path, "training_labels.txt"), "w") as f:
        for label in labels[train_idx]:
            f.write(f"{label}\n")


def main():
    # Parse arguments
    type_data, input_folder_path, output_folder_path = parse_args()
    text, labels = extract_text_and_labels(type_data, input_folder_path)
    # Train, val, test split
    train_idx, val_idx, test_idx = get_train_test_index(
        input_size=text.size,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        random_state=42,
    )
    # Write output training, validation and testing
    write_to_output(text, labels, train_idx, val_idx, test_idx, output_folder_path)


if __name__ == "__main__":
    main()
