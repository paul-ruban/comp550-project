import os

from pathlib import Path
from src.utils.compressor import Compressor
from src.utils.decompressor import Decompressor
from src.utils.file_metrics import FileMetrics

"""
    Script to test the file utils of compression, decompression and file size.
    Remove test_compression_data.txt.gz and test_decompression_data.txt to
    fully test.
"""


def main():
    os.curdir = os.path.dirname(os.path.abspath(__file__))
    # Test compression
    data_path = Path(os.path.join(os.curdir, "test_data", "test_compression_data.txt"))
    compressor = Compressor(data_path=data_path)
    compressor.compress()
    # Test decompression
    compressed_data_path = compressor.get_compressed_data_path()
    decompressor = Decompressor(data_path=compressed_data_path)
    decompressed_data_path = decompressor.decompress(new_unzipped_file_name="test_decompression_data.txt")
    # Test file metrics
    # Original data path
    file_metrics = FileMetrics(data_path)
    print(f"Size (in bits) of original data {data_path.name}: {file_metrics.get_file_size()}")
    print(
        f"Size (in human readable format) of original data {data_path.name}: {file_metrics.get_human_readable_size()}"
    )
    # Compressed data path
    file_metrics = FileMetrics(compressed_data_path)
    print(f"Size (in bits) of compressed data {data_path.name}: {file_metrics.get_file_size()}")
    print(
        f"Size (in human readable format) of compressed data {data_path.name}: {file_metrics.get_human_readable_size()}"
    )
    # Decompressed data path
    file_metrics = FileMetrics(decompressed_data_path)
    print(f"Size (in bits) of decompressed data {data_path.name}: {file_metrics.get_file_size()}")
    print(
        f"Size (in human readable format) of decompressed data {data_path.name}: {file_metrics.get_human_readable_size()}"
    )


if __name__ == "__main__":
    main()
