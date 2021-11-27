import copy
import os
import random

from abc import abstractmethod
from typing import List
from collections import Counter
import numpy as np
from typing import Tuple, Union
from src.utils.compressor import Compressor
from src.utils.eval_metrics import compression_accuracy


class Mask:
    def __init__(self, mask_token) -> None:
        self.mask_token = mask_token

    @abstractmethod
    def mask(self, X: List[List[str]]) -> List[List[str]]:
        """X is a list of lists of strings, a string represents one token"""
        pass

    def compression_score(
        self,
        X_original: Union[List[str], List[List[str]]],
        X_masked: Union[List[str], List[List[str]]],
    ) -> float:
        """
        X_original is a list of strings, a string represents one text
        X_masked is a list of strings, a string represents one text
        if either are passed as lists of tokens, then they are joined by spaces

        Compresses each file with gzip, gets the compression score and returns the average
        """
        assert len(X_original) == len(
            X_masked
        ), "Original and masked lists must have the same length."
        if isinstance(X_original[0], list):
            X_original = [" ".join(x) for x in X_original]
        if isinstance(X_masked[0], list):
            X_masked = [" ".join(x) for x in X_original]
        temp_data_path = os.path.join(os.getcwd(), "..", "..", "data", "temp")
        compression_scores = np.zeros(len(X_original))
        for i, (x_original, x_masked) in enumerate(zip(X_original, X_masked)):
            original_file_path = os.path.join(temp_data_path, "original.txt")
            masked_file_path = os.path.join(temp_data_path, "masked.txt")
            with open(original_file_path, "w") as f:
                f.write(x_original)
            with open(masked_file_path, "w") as f:
                f.write(x_masked)
            compressor = Compressor(data_path=masked_file_path)
            compressor.compress()
            compressed_masked_file_path = compressor.get_compressed_data_path()
            compression_scores[i] = compression_accuracy(
                original_file_path=original_file_path,
                compressed_file_path=compressed_masked_file_path,
            )
        return compression_scores.mean()


class RandomMask(Mask):
    """Implements random masking strategy."""

    def __init__(self, proba, mask_token, adjust_by_length=False) -> None:
        assert 0 < proba <= 1, "Masking probability proba must be > 0 and <= 1"
        self.proba = proba
        self.adjust_by_length = adjust_by_length
        self.mask_token = mask_token

    def mask(self, X: List[List[str]]) -> List[List[str]]:
        """X is a list of lists of strings, a string represents one token"""
        # make a copy of the data not to mess up the orogonal
        _X = copy.deepcopy(X)
        for i in range(len(_X)):
            for j in range(len(i)):
                proba = self.proba
                # the probability of masking is adjusted by the token length, increasing probability of longer tokens
                # Ex.
                # length:       1     2      3      4      5      6      7      8      9      10
                # ajdustment: [1.0, 1.044, 1.071, 1.091, 1.106, 1.118, 1.129, 1.139, 1.147, 1.155]
                if self.adjust_by_length:
                    proba *= len(_X[i][j]) ** (
                        1 / 2 ** 4
                    )  # adjustment formula can be tuned if needed
                if random.random() < proba:
                    _X[i][j] = self.mask_token

        return _X


class LengthBasedMask(Mask):
    def __init__(self, ratio, strategy, mask_token) -> None:
        assert 0 < ratio <= 1, "Masking ratio must be > 0 and <= 1"
        assert strategy in ["sentence", "all"], "Masking strategy must be one of: [sent, all]"
        self.ratio = ratio
        self.strategy = strategy
        self.mask_token = mask_token

    def mask(self, X: List[List[str]]) -> List[List[str]]:
        """X is a list of lists of strings, a string represents one token"""
        # make a copy of the data not to mess up the orogonal
        _X = copy.deepcopy(X)

        if self.strategy == "all":
            min_masked_len, max_masked_count = self.get_masked_min_len_max_count(
                X=_X, ratio=self.ratio
            )
            masked = 0

        for i in range(len(_X)):
            if self.strategy == "sentence":
                min_masked_len, max_masked_count = self.get_masked_min_len_max_count(
                    X=_X[i], ratio=self.ratio
                )
            for j in range(len(i)):
                if len(_X[i][j]) >= min_masked_len and masked < max_masked_count:
                    _X[i][j] = self.mask_token
                    masked += 1
            if self.strategy == "sentence":
                masked = 0

        return _X

    def get_masked_min_len_max_count(X: List[List[str]], ratio: float) -> Tuple[int, int]:
        if any(isinstance(x, list) for x in X):
            # X is a List[List[str]], need to flatten it
            X = [t for x in X for t in x]

        max_masked_count = int(len(X) * ratio)
        min_masked_len = max(1, sorted([len(t) for t in X], reverse=True)[max_masked_count - 1])

        return min_masked_len, max_masked_count


class RandomWindowMask(Mask):
    def __int__(
        self,
        mask_token: str = "_",
        window_size: int = 10,
        num_of_masks: int = 1,
        prop_masked: float = None,
        random_seed: int = 42,
    ) -> None:
        assert (num_of_masks is not None and prop_masked is None) or (
            num_of_masks is None and prop_masked is not None
        ), "You must be either a fixed num of tokens to mask within a window or a proportion."
        self.mask_token = mask_token
        self.window_size = window_size
        self.num_of_masks = num_of_masks
        self.prop_masked = prop_masked
        self.random_seed = random_seed

    def mask(self, X: Union[List[str], List[List[str]]]) -> Union[List[str], List[List[str]]]:
        """Expects X to be a list of strings (where the string is a text)
        or a list of list of strings where each sub list is the list of
        all tokens of a text.
        """
        random.seed(self.random_seed)
        _X = copy.deepcopy(X)
        _X = [x.split() if isinstance(x, str) else x for x in _X]
        masked_X = []
        for tokenized_text in _X:
            num_of_masks = (
                int(self.window_size * self.prop_masked)
                if self.prop_masked is not None
                else self.num_of_masks
            )
            # Create the indices to be masked, do this for each window
            masked_indices = [
                j
                for i in range(int(len(tokenized_text) / self.window_size))
                for j in random.sample(
                    range(i * self.window_size, (i + 1) * self.window_size), num_of_masks
                )
            ]
            # Mask at the indices
            masked_text = [
                self.mask_token if i in masked_indices else token
                for i, token in enumerate(tokenized_text)
            ]
            # If used a list of strings as X, convert it back
            masked_text = " ".join(masked_text) if isinstance(X[0], str) else masked_text
            masked_X.append(masked_text)
        return masked_X


class LengthWindowMask(Mask):
    def __init__(
        self,
        mask_token: str = "_",
        window_size: int = 10,
        num_of_masks: int = 1,
        prop_masked: float = None,
    ):
        assert (num_of_masks is not None and prop_masked is None) or (
            num_of_masks is None and prop_masked is not None
        )
        self.mask_token = mask_token
        self.window_size = window_size
        self.num_of_masks = num_of_masks
        self.prop_masked = prop_masked

    def mask(self, X: Union[List[str], List[List[str]]]) -> Union[List[str], List[List[str]]]:
        """Expects X to be a list of strings (where the string is a text)
        or a list of list of strings where each sub list is the list of
        all tokens of a text.
        """
        masked_X = []
        _X = copy.deepcopy(X)
        _X = [x.split() if isinstance(x, str) else x for x in _X]
        for tokenized_text in _X:
            tokenized_text_with_index = list(enumerate(tokenized_text))
            num_of_masks = (
                int(self.window_size * self.prop_masked)
                if self.prop_masked is not None
                else self.num_of_masks
            )
            # Get the indices that correspond to the longest words within a window
            # These will be used to mask
            masked_indices = [
                j
                for i in range(int(len(tokenized_text_with_index) / self.window_size))
                for (j, _) in sorted(
                    tokenized_text_with_index[i * self.window_size : (i + 1) * self.window_size],
                    key=lambda x: len(x[1]),
                    reverse=True,
                )[:num_of_masks]
            ]
            masked_text = [
                self.masking_token if i in masked_indices else token
                for i, token in tokenized_text_with_index
            ]
            # If used a list of strings as X, convert it back
            masked_text = " ".join(masked_text) if isinstance(X[0], str) else masked_text
            masked_X.append(masked_text)
        return masked_X


class FrequencyBasedMask(Mask):
    def __init__(self, top_freq, mask_token) -> None:
        assert top_freq > 0, "Top frequent must be positive"
        self.top_freq = top_freq
        self.mask_token = mask_token

    def mask(self, X: Union[List[str], List[List[str]]]) -> Union[List[str], List[List[str]]]:
        """X is a list of lists of strings, a string represents one token
        Can also be a list of strings in which case the method does an extra tokenization
        step"""
        # make a copy of the data not to mess up the original
        _X = copy.deepcopy(X)
        _X = [x.split() if isinstance(x, str) else x for x in _X]

        tokens_to_mask = self.get_most_frequent(X=_X, top_freq=self.top_freq)

        for i in range(len(_X)):
            for j in range(len(i)):
                if _X[i][j] in tokens_to_mask:
                    _X[i][j] = self.mask_token

        # If used a list of strings as X, convert it back to this form
        _X = [" ".join(x) for x in _X] if isinstance(X[0], str) else _X
        return _X

    def get_most_frequent(X, top_freq) -> List[str]:
        # flatten data
        X = [t for x in X for t in x]
        most_frequent = Counter(X).most_common(top_freq)
        tokens_to_mask = [x[0] for x in most_frequent]

        return tokens_to_mask


class ModifiedTfidMask(Mask):
    def __init__(self, mask_token) -> None:
        super().__init__(mask_token)

    def mask(self, X: Union[List[str], List[List[str]]]) -> Union[List[str], List[List[str]]]:
        pass


class StopwordMask(Mask):
    def __init__(
        self,
        ratio: float,
        strategy: str,
        mask_token: str = "_",
        window_size: int = 10,
    ):
        pass

    def mask(self, X: List[List[str]]) -> List[List[str]]:
        pass


class POSMask(Mask):
    def __init__(
        self,
        ratio: float,
        strategy: str,
        mask_token: str = "_",
        window_size: int = 10,
    ):
        pass

    def mask(self, X: List[List[str]]) -> List[List[str]]:
        pass


class ProbabilisticMask(Mask):
    def __init__(self, mask_probas, mask_token) -> None:
        self.mask_probas = mask_probas
        self.mask_token = mask_token
