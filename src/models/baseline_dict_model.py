from string import printable
from nltk import ngrams, FreqDist
from typing import List

from nltk.tokenize import word_tokenize


class BaselineDictModel:
    def __init__(self, number_of_types_to_replace: int = 0) -> None:
        assert (
            isinstance(number_of_types_to_replace, int) and number_of_types_to_replace >= 0
        ), "number_of_types_to_replace must be an integer greater than or equal to 0"
        if number_of_types_to_replace >= len(printable):
            number_of_types_to_replace = len(printable)
        self.number_of_types_to_replace = number_of_types_to_replace

    def fit(self, X: List[str]) -> None:
        # Create single string from list of strings
        input_string = (" ".join(X)).split()
        # Tokenize into n-gram
        self.freq_dict = FreqDist(ngrams(input_string, 1))

    def encode(self, x: str) -> str:
        # Encode using most frequent token replacement
        input_string = x.split()
        if self.freq_dict is None:
            # Use the frequency of the string itself
            n_gram = ngrams(input_string, 1)
            freq_dict = FreqDist(n_gram)
        else:
            freq_dict = self.freq_dict
        assigned_dict = {}
        # Replace n-grams with most frequent n-gram
        i = 0
        for ((word_type,), _) in freq_dict.most_common():
            if word_type in input_string and i < self.number_of_types_to_replace:
                input_string = [
                    printable[i] if word == word_type else word for word in input_string
                ]
                assigned_dict[word_type] = printable[i]
                i += 1
        self.assigned_dict = assigned_dict
        return " ".join(input_string)

    def decode(self, x: str) -> str:
        assert (
            self.assigned_dict is not None
        ), "Must encode the model before decoding"
        # Decode using the inverse process
        input_string = x.split()
        # Replace encoded characters back
        for word_type, encoded_char in self.assigned_dict.items():
            input_string = [word_type if word == encoded_char else word for word in input_string]
        return " ".join(input_string)


if __name__ == "__main__":
    X = ["i like to shop", "he likes to shop", "we all like to shop"]
    model = BaselineDictModel(number_of_types_to_replace=5)
    model.fit(X)
    encoded_string = model.encode("i like jake and he likes shopping")
    print(encoded_string)
    decoded_string = model.decode(encoded_string)
    print(decoded_string)
