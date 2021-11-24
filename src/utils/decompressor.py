import gzip

from pathlib import Path


class Decompressor:
    def __init__(self, data_path: str) -> None:
        """Decompressor class. Decompresses data in a .gz zip.

        Args:
            data_path (str): Absolute path of txt.gz file.
        """
        data_path = Path(data_path)
        assert data_path.exists(), f"Data path {data_path} does not exist."
        assert "".join(data_path.suffixes) == ".txt.gz", f"Data path {data_path} must be a .txt.gz file."
        self.data_path = data_path

    def decompress(self, new_unzipped_file_name: str) -> Path:
        """
        Decompresses data in text file with gzip using original path and
        new unzipped .txt file name. Returns the path of the decompressed file.
        """
        new_unzipped_file_name = Path(new_unzipped_file_name)
        assert (
            new_unzipped_file_name.suffix == ".txt"
        ), f"New unzipped filename {new_unzipped_file_name} must be a text file."
        decompressed_data_path = self.data_path.parent / new_unzipped_file_name.name
        # Decompress the data
        with gzip.open(self.data_path, "rb") as f:
            decompressed_file = f.read()
            with open(decompressed_data_path, "w") as f_out:
                f_out.write(decompressed_file.decode("utf-8"))
        return decompressed_data_path
