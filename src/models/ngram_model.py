import copy
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import List, Union

import dill as pickle
import nltk.lm
import numpy as np
from nltk.lm.preprocessing import (flatten, pad_both_ends,
                                   padded_everygram_pipeline)
from nltk.util import ngrams
from src.models.model import Model


class NGramModel(Model):
    def __init__(self, n):
        self.n = n
        self.lm = None

    def fit(self, X: List[str], lm: nltk.lm, **kwargs) -> nltk.lm:
        # This assumption of padding is reasonable because we are in some
        # sense modeling at the document level so <s> and </s> represent
        # the start and end token for the n-gram model (they have a slightly
        # different interpretation for RNNs and Bert)
        X = [x.split() for x in X]
        train, vocab = padded_everygram_pipeline(self.n, X)
        self.lm = lm(order=self.n, **kwargs)
        self.lm.fit(train, vocab)
        return self.lm

    def preprocess(self, X: List[str]) -> List[str]:
        """
        Preprocess the text to be used in the model.
        """
        train = [x.split() for x in X]
        X = [pad_both_ends(x, n=self.n) for x in X]
        X = [ngrams(x, n=self.n) for x in X]
        vocab = set(flatten(X))
        return train, vocab

    def load_model(self, pickle_model_path: str) -> nltk.lm:
        pickle_model_path = Path(pickle_model_path)
        assert pickle_model_path.suffix == ".pkl", "Must be a pickle file"
        assert pickle_model_path.exists(), "Model file does not exist"
        with open(pickle_model_path, "rb") as f:
            self.lm = pickle.load(f)
        return self.lm

    def save_model(self, pickle_model_path: str) -> None:
        pickle_model_path = Path(pickle_model_path)
        assert pickle_model_path.suffix == ".pkl", "Must be a pickle file"
        assert self.lm is not None, "Must fit the model before saving"
        with open(pickle_model_path, "wb") as f:
            pickle.dump(self.lm, f)

    def average_perplexity(self, X: Union[List[str], List[List[str]]]) -> float:
        if self.lm is None:
            raise ValueError("Must fit the model before computing average perplexity")
        else:
            _X = copy.deepcopy(X)
            _X = _X if isinstance(X[0], list) else [x.split() for x in _X]
            _X = [ngrams(pad_both_ends(x, n=self.n), self.n) for x in _X]
            perplexities = np.array([self.lm.perplexity(x) for x in _X])
            return perplexities.mean()

    def decode(
        self,
        X: Union[List[str], List[List[str]]],
        masking_char: str = "_",
        random_seed: int = 42,
        parallel: bool = False,
    ) -> List[str]:
        """
        Decode the masked text x to its original form.
        """
        if self.lm is None:
            raise ValueError("Must fit the model before decoding")
        else:
            X_encoded = copy.deepcopy(X)
            X_encoded = X_encoded if isinstance(X[0], list) else [x.split() for x in X_encoded]
            X_decoded = []
            if self.n == 1:
                # In this case it's more appropriate to generate ALL at the same time
                num_words_to_decode = sum(
                    [sum([int(x == masking_char) for x in x_encoded]) for x_encoded in X_encoded]
                )
                # lm_copy = copy.deepcopy(self.lm)
                # lm_copy.vocab.counts = {k: v for k, v in self.lm.vocab.counts.items() if v > 5}
                generated_tokens = self.l.generate(
                    num_words=num_words_to_decode, random_seed=random_seed, text_seed=None
                )
                X_decoded = [
                    [generated_tokens.pop() if x == masking_char else x for x in x_encoded]
                    for x_encoded in X_encoded
                ]
                X_decoded = (
                    X_decoded if isinstance(X[0], list) else [" ".join(x) for x in X_decoded]
                )
            else:
                if parallel:
                    with Pool() as pool:
                        X_decoded = pool.map(
                            partial(
                                self._decode,
                                masking_char=masking_char,
                                random_seed=random_seed,
                            ),
                            X_encoded,
                        )
                else:
                    for x in X_encoded:
                        x_decoded = self._decode(x, masking_char)
                        X_decoded.append(x_decoded)
            return X_decoded

    def _decode(
        self, x: Union[str, List[str]], masking_char: str = "_", random_seed: int = 42
    ) -> Union[str, List[str]]:
        x_decoded = list(pad_both_ends(x, n=self.n))
        indices_to_decode = [i for i, x in enumerate(x_decoded) if x == masking_char]
        for i in indices_to_decode:
            context = x_decoded[i - self.n + 1 : i]
            x_decoded[i] = self.lm.generate(num_words=1, random_seed=random_seed, text_seed=context)
        # Remove the padding
        x_decoded = x_decoded[self.n - 1 : -self.n + 1] if self.n > 1 else x_decoded
        x_decoded = x_decoded if isinstance(x, list) else " ".join(x_decoded)
        return x_decoded
