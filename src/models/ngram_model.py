import copy
from pathlib import Path
from typing import List, Union

import dill as pickle
import nltk.lm
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline
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

    def decode(self, X: Union[List[str], List[List[str]]], masking_char: str = "_") -> List[str]:
        """
        Decode the masked text x to its original form.
        """
        if self.lm is None:
            raise ValueError("Must fit the model before decoding")
        else:
            # The text has already been tokenized and separated by spaces.
            X_encoded = copy.deepcopy(X)
            X_encoded = (
                X_encoded if isinstance(X[0], list) else [x.split() for x in X_encoded]
            )
            X_decoded = []
            for x in X_encoded:
                # Pad the very beginning and end with <s> and </s> tokens.
                # To satisfy probability distribution.
                x_decoded = list(pad_both_ends(x, n=self.n))
                indices_to_decode = [i for i, x in enumerate(x_decoded) if x == masking_char]
                for i in indices_to_decode:
                    context = x_decoded[i - self.n + 1 : i] if self.n > 1 else []
                    x_decoded[i] = self.lm.generate(num_words=1, text_seed=context)
                # Remove the padding
                x_decoded = x_decoded[self.n - 1 : -self.n + 1] if self.n > 1 else x_decoded
                x_decoded = x_decoded if isinstance(X[0], list) else " ".join(x_decoded)
                X_decoded.append(x_decoded)
            # Return the decoded text as a string
            return x_decoded
