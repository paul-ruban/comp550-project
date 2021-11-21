import os
from pathlib import Path
from src.models.baseline_dict_model import BaselineDictModel
from src.utils.compressor import Compressor
from src.utils.decompressor import Decompressor
from src.utils.eval_metrics import compression_accuracy, edit_distance_accuracy, harmonic_mean_score


"""
    Script to test the eval metrics utils and how it can be used with
    one of our models.
"""


def main():
    os.curdir = os.path.dirname(os.path.abspath(__file__))
    # TODO: Need methods for 1 and 2 large scale batches
    # 1. Raw input text
    data_path = Path(os.path.join(os.curdir, "test_data", "test_compression_data.txt"))
    # 2. Preprocessed data
    with open(data_path, "r") as f:
        input_string = f.read()
    X_train = [input_string.replace("\n", " ").strip()]
    # 3. Train model and encode input
    baseline_dict_model = BaselineDictModel(number_of_types_to_replace=1)
    baseline_dict_model.fit(X_train)
    x_encoded = baseline_dict_model.encode(x=X_train[0])
    # 4. Pass encoded text to compression module and compress
    encoded_path = Path(os.path.join(os.curdir, "test_data", "test_compression_data_encoded.txt"))
    with open(encoded_path, "w") as f:
        f.write(x_encoded)
    compressor = Compressor(data_path=encoded_path)
    compressor.compress()
    compressed_path = compressor.get_compressed_data_path()
    # 5. Decompress and decode text
    decompressor = Decompressor(data_path=compressed_path)
    decompressed_path = decompressor.decompress(new_unzipped_file_name="test_decompression_data_encoded.txt")
    with open(decompressed_path, "r") as f:
        decompressed_string = f.read()
    x_decoded = baseline_dict_model.decode(x=decompressed_string)
    # TODO: Module for reporting metrics
    # 6. Report metrics
    print(f"Compression accuracy: {compression_accuracy(data_path, compressed_path)}")
    print(f"Edit distance accuracy: {edit_distance_accuracy(X_train[0], x_decoded)}")
    print(
        f"Harmonic mean score: {harmonic_mean_score(data_path, compressed_path, X_train[0], x_decoded)}"
    )


if __name__ == "__main__":
    main()
