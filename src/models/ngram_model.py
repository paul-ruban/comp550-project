import random
from pathlib import Path
from typing import List

import dill as pickle
import nltk.lm
from nltk.lm.preprocessing import pad_both_ends
from nltk.lm.preprocessing import padded_everygram_pipeline


class NGramModel:
    CODING_CHAR = "A"

    def __init__(self, n):
        self.n = n
        self.lm = None

    def fit(self, X: List[List[str]], lm: nltk.lm, **kwargs) -> nltk.lm:
        train, vocab = padded_everygram_pipeline(self.n, X)
        self.lm = lm(**kwargs)
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

    def encode(self, x: List[str], proportion: float, encoding_type: str = "random", seed: int = 42) -> List[str]:
        if encoding_type == "random":
            num_words_to_encode = int(len(x) * proportion)
            random.seed(seed)
            replacement_index = random.sample(range(len(x)), num_words_to_encode)
            x_encoded = [
                NGramModel.CODING_CHAR if i in replacement_index else x[i] for i in range(len(x))
            ]
            return x_encoded
        elif encoding_type == "greedy":
            num_words_to_encode = int(len(x) * proportion)
            # get index by largest character word
            replacement_index = sorted(range(len(x)), key=lambda i: len(x[i]), reverse=True)[
                :num_words_to_encode
            ]
            x_encoded = [
                NGramModel.CODING_CHAR if i in replacement_index else x[i] for i in range(len(x))
            ]
        else:
            raise ValueError("No other encoding is supported.")

    def decode(self, x: List[str], random_seed: int = 42) -> List[str]:
        if self.lm is None:
            raise ValueError("Must fit the model before decoding")
        else:
            x_decoded = list(pad_both_ends(x, n=self.n))
            for i, word in enumerate(x_decoded):
                if word == NGramModel.CODING_CHAR:
                    context = x_decoded[i - self.n + 1 : i] if self.n > 1 else []
                    x_decoded[i] = self.lm.generate(
                        num_words=1, random_seed=random_seed, text_seed=context
                    )
            x_decoded = x_decoded[self.n - 1 : -self.n + 1] if self.n > 1 else x_decoded
            return x_decoded
