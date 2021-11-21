from abc import abstractmethod
from typing import List


class Model:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self, X: List[str]) -> None:
        """X is a list of strings, a string represents one text"""
        pass

    @abstractmethod
    def encode(self, x: str) -> str:
        """encode string x, the string x represents one text,
        this is like part 1 of predict."""
        pass

    @abstractmethod
    def decode(self, x: str) -> str:
        """decode string x, the string x represents one encoded text,
        this is like part 2 of predict"""
        pass
