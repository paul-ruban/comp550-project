import gzip
import shutil

from pathlib import Path


class Compressor:
    def __init__(self, data_path: str) -> None:
        """Compressor class. Compresses data in text file with gzip.

        Args:
            data_path (str): Absolute path of text file.
        """
        data_path = Path(data_path)
        assert data_path.exists(), f"Data path {data_path} does not exist."
        assert data_path.suffix == ".txt", f"Data path {data_path} must be a text file."
        self.data_path = data_path

    def get_compressed_data_path(self) -> Path:
        """
        Returns the path of the compressed data.
        """
        return self.data_path.parent / (self.data_path.name + ".gz")

    def compress(self) -> None:
        """
        Compresses data in text file with gzip uses original path and affixes it with .gz.
        """
        compressed_data_path = self.get_compressed_data_path()
        # Compress the data
        with open(self.data_path, "rb") as f_in:
            with gzip.open(compressed_data_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)