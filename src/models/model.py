import numpy as np
from abc import abstractmethod
from typing import List, Union


class Model:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self, X: List[str]) -> None:
        """X is a list of strings, a string represents one text"""
        pass

    @abstractmethod
    def decode(self, x: str) -> str:
        """decode string x, the string x represents one encoded text,
        this is like part 2 of predict"""
        pass

    def accuracy_score(
        self,
        masking_token: str,
        X_original: Union[List[str], List[List[str]]],
        X_masked: Union[List[str], List[List[str]]],
        X_decoded: Union[List[str], List[List[str]]],
    ) -> float:
        """
        X_original is a list of strings, a string represents one text
        X_decoded is a list of strings, a string represents one text
        if either are passed as lists of strings, then they are tokenized

        Finds the average reconstruction accuracy, focuses explicitly on
        the tokens that have been masked.
        """
        if isinstance(X_original[0], str):
            X_original = [x.split() for x in X_original]
        if isinstance(X_masked[0], str):
            X_masked = [x.split() for x in X_masked]
        if isinstance(X_decoded[0], str):
            X_decoded = [x.split() for x in X_decoded]
        assert len(X_original) == len(X_masked) == len(X_decoded) and all(
            len(x_og) == len(x_mask) == len(x_dec)
            for x_og, x_mask, x_dec in zip(X_original, X_masked, X_decoded)
        ), "The outer or inner length of your original and decoded tokens is different."
        reconstruction_accuracy = np.zeros(len(X_original))
        for i, (x_original, x_masked, x_decoded) in enumerate(zip(X_original, X_masked, X_decoded)):
            masked_results = [
                int(x_o == x_d)
                for x_o, x_m, x_d in zip(x_original, x_masked, x_decoded)
                if x_m == masking_token
            ]
            reconstruction_accuracy[i] = sum(masked_results) / len(masked_results)
        return np.mean(reconstruction_accuracy)

    def similarity_score(
        self,
        masking_token: str,
        X_original: Union[List[str], List[List[str]]],
        X_masked: Union[List[str], List[List[str]]],
        X_decoded: Union[List[str], List[List[str]]],
    ) -> float:
        """
        X_original is a list of strings, a string represents one text
        X_decoded is a list of strings, a string represents one text
        if either are passed as lists of strings, then they are tokenized

        Finds the average similarity score, focuses explicitly on
        the tokens that have been masked.
        """
        if isinstance(X_original[0], str):
            X_original = [x.split() for x in X_original]
        if isinstance(X_masked[0], str):
            X_masked = [x.split() for x in X_masked]
        if isinstance(X_decoded[0], str):
            X_decoded = [x.split() for x in X_decoded]
        assert len(X_original) == len(X_masked) == len(X_decoded) and all(
            len(x_og) == len(x_mask) == len(x_dec)
            for x_og, x_mask, x_dec in zip(X_original, X_masked, X_decoded)
        ), "The outer or inner length of your original and decoded tokens is different."
        similarity_score = np.zeros(len(X_original))
        for i, (x_original, x_masked, x_decoded) in enumerate(zip(X_original, X_masked, X_decoded)):
            masked_results = [
                measure_similarity(x_o, x_d)
                for x_o, x_m, x_d in zip(x_original, x_masked, x_decoded)
                if x_m == masking_token
            ]
            similarity_score[i] = np.mean(masked_results)
        return np.mean(similarity_score)
