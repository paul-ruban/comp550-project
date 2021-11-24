import editdistance
import os

from pathlib import Path
from src.utils.compressor import Compressor
from src.utils.file_metrics import FileMetrics


def compression_ratio(original_file_path: str, compressed_file_path: str) -> float:
    """
    Computes the compression ratio of a file. A CR of 1 means that the compressed
    file has the same size of the original file. We want to reduce the CR as much as
    possible.
    """
    file_metrics_original = FileMetrics(file_path=original_file_path)
    file_metrics_compressed = FileMetrics(file_path=compressed_file_path)
    original_file_size = file_metrics_original.get_file_size()
    compressed_file_size = file_metrics_compressed.get_file_size()
    return compressed_file_size / original_file_size


def compression_accuracy(original_file_path: str, compressed_file_path: str) -> float:
    """This is 1 - CR i.e. we want to get this value as close to 1 as possible"""
    return 1 - compression_ratio(original_file_path, compressed_file_path)


def edit_distance(original_string: str, decompressed_string: str) -> float:
    """Finds the edit distance between the original and uncompressed text.
    Since we want the edit distance of words we convert the strings to list
    of strings.
    """
    original_text = original_string.split()
    decompressed_text = decompressed_string.split()
    return editdistance.eval(original_text, decompressed_text)


def edit_distance_accuracy(original_string: str, uncompressed_string: str) -> float:
    """Finds the *relative* edit distance between the original and compressed text.
    To create a relative score we divide the edit distance by
    max(size_original, size_compressed) - |size_original - size_compressed| which
    are respectively upper and lower bounds to the edit distance. To convert this to
    an accuracy score we take 1 - relative edit distance.
    """
    # Tokenize
    original_text = original_string.split()
    uncompressed_text = uncompressed_string.split()
    # Find max and min edit distance
    min_edit_distance = abs(len(original_text) - len(uncompressed_text))
    max_edit_distance = max(len(original_text), len(uncompressed_text))
    # Relative edit distance
    relative_edit_distance = edit_distance(original_string, uncompressed_string) / (
        max_edit_distance - min_edit_distance
    )
    return 1 - relative_edit_distance


def harmonic_mean_score(
    original_file_path: str,
    compressed_file_path: str,
    original_string: str,
    uncompressed_string: str,
) -> float:
    """This is the harmonic mean of the edit distance accuracy and the compression accuracy.
    This is similar to the F1 score or the V-measure score where we have two measure which
    we would like to both maximize. Instead of taking their mean we take their harmonic mean
    to have a more conservative score.
    """
    compression_acc = compression_accuracy(original_file_path, compressed_file_path)
    edit_distance_acc = edit_distance_accuracy(original_string, uncompressed_string)
    return 2 / (1 / compression_acc + 1 / edit_distance_acc)


if __name__ == "__main__":
    os.curdir = os.path.dirname(os.path.abspath(__file__))
    # Test
    data_path = Path(
        os.path.join(os.curdir, "..", "..", "tests", "test_data", "test_compression_data.txt")
    )
    compressor = Compressor(data_path=data_path)
    compressed_file_path = compressor.get_compressed_data_path()
    print(f"Compression accuracy: {compression_accuracy(data_path, compressed_file_path)}")
    print(
        f"Edit distance accuracy: {edit_distance_accuracy('i like tomatoes a lot', 'i enjoy tomatoes a bit bit')}"
    )
    print(
        f"Harmonic mean score: {harmonic_mean_score(data_path, compressed_file_path, 'i like tomatoes a lot', 'i enjoy tomatoes a bit bit')}"
    )
