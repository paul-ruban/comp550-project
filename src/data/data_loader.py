from typing import List
from pathlib import Path


def load_data(data_path: str) -> List[str]:
    """
    Loads data from the given path as list of strings. Each
    string is a text.
    """
    data_path = Path(data_path)
    file_paths = sorted(data_path.glob("**/*.txt"))
    data = []
    for file_path in file_paths:
        with open(file_path, "r") as f:
            data.append(f.read())
    return data
