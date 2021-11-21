import os

from pathlib import Path


class FileMetrics:
    def __init__(self, file_path: str) -> None:
        """Computes various file metrics.

        Args:
            file_path (str): Path of file (expects absolute path i.e.
            the user must handle relative paths before calling this class).
        """
        file_path = Path(file_path)
        assert file_path.exists()
        self.file_path = file_path

    def get_file_size(self) -> int:
        """
        Get the size of the file in number of bits.
        """
        return os.path.getsize(self.file_path)

    def get_human_readable_size(self) -> str:
        """
        Get the size of the file in human readable format.
        """
        size_in_bits = self.get_file_size()
        quantifier = ["", "Ki", "Mi", "Gi"]
        for unit in quantifier:
            if size_in_bits < 1024:
                return f"{size_in_bits:.3f} {unit}B"
            size_in_bits /= 1024
        return f"{size_in_bits} TiB"
