import string.printable

from nltk import ngrams, FreqDist
from type import List


class BaselineDictModel:
    def __init__(self, n_gram: int = 1, number_of_types_to_replace: int = 0) -> None:
        assert isinstance(n_gram, int) and n_gram >= 1, "n_gram must be an integer greater than or equal to 1"
        assert isinstance(number_of_types_to_replace, int) and number_of_types_to_replace >= 0, "number_of_types_to_replace must be an integer greater than or equal to 0"
        if number_of_types_to_replace >= len(string.printable):
            number_of_types_to_replace = len(string.printable)
        self.number_of_types_to_replace = number_of_types_to_replace
        self.n_gram = n_gram

    def fit(self, X: List[str]) -> None:
        # Create single string from list of strings
        input_string = (" ".join(X)).split()
        # Tokenize into n-gram
        self.freq_dict = FreqDist(ngrams(input_string, self.n_gram))

    def encode(self, x: str) -> str:
        # Encode with masks
        # Tokenize into n-gram
        input_string = x.split()
        # Replace n-grams with most frequent n-gram

        # Add logic if dict is missing

    def decode(self, x: str) -> str:
        # Use dict to decode

