import copy
import random
from abc import abstractmethod
from typing import List, Tuple
from collections import Counter


class Mask:
    def __init__(self, mask_token) -> None:
        self.mask_token = mask_token

    @abstractmethod
    def mask(self, X: List[List[str]]) -> List[List[str]]:
        """X is a list of lists of strings, a string represents one token"""
        pass


class RandomMask(Mask):
    """Implements random masking strategy."""
    def __init__(self, proba, mask_token, adjust_by_length=False) -> None:
        assert(0 < proba <= 1), "Masking probability proba must be > 0 and <= 1"
        self.proba = proba
        self.adjust_by_length = adjust_by_length
        self.mask_token = mask_token
    
    def mask(self, X: List[List[str]]) -> List[List[str]]:
        """X is a list of lists of strings, a string represents one token"""
        # make a copy of the data not to mess up the orogonal
        _X = copy.deepcopy(X)
        for i in range(len(_X)):
            for j in range(len(_X[i])):
                proba = self.proba
                # the probability of masking is adjusted by the token length, increasing probability of longer tokens
                # Ex.
                # length:       1     2      3      4      5      6      7      8      9      10
                # ajdustment: [1.0, 1.044, 1.071, 1.091, 1.106, 1.118, 1.129, 1.139, 1.147, 1.155]
                if self.adjust_by_length:
                    proba *= len(_X[i][j])**(1/2**4) # adjustment formula can be tuned if needed
                if random.random() < proba:
                    _X[i][j] = self.mask_token

        return _X


class LengthBasedMask(Mask):
    def __init__(self, ratio, strategy, mask_token) -> None:
        assert(0 < ratio <= 1), "Masking ratio must be > 0 and <= 1"
        assert(strategy in ["sentence", "all"]), "Masking strategy must be one of: [sent, all]"
        self.ratio = ratio
        self.strategy = strategy
        self.mask_token = mask_token
    
    # Takes into account randomly masking equally size tokens
    def mask(self, X: List[List[str]]) -> List[List[str]]:
        """X is a list of lists of strings, a string represents one token"""
        # make a copy of the data not to mess up the orogonal
        _X = copy.deepcopy(X)

        if self.strategy == "all":
            min_masked_len, max_masked_count = self.get_masked_min_len_max_count(X=_X, ratio=self.ratio)
            masked = 0

        for i in range(len(_X)):
            if self.strategy == "sentence":
                min_masked_len, max_masked_count = self.get_masked_min_len_max_count(X=[_X[i]], ratio=self.ratio)
                masked = 0

            maskable_token_ids = {}

            for idx, token in enumerate(_X[i]):
                if len(token) >= min_masked_len:
                    if (len(token) in maskable_token_ids):
                        maskable_token_ids[len(token)].append(idx)
                    else:
                        maskable_token_ids[len(token)] = [idx]

            for k in maskable_token_ids:
                random.shuffle(maskable_token_ids[k])

            while (masked < max_masked_count and maskable_token_ids):
                len_to_mask = max(maskable_token_ids.keys())
                id_to_mask = maskable_token_ids[len_to_mask].pop() if len(maskable_token_ids[len_to_mask]) > 1 else maskable_token_ids.pop(len_to_mask)[0]
                
                _X[i][id_to_mask] = self.mask_token
                masked += 1

            if self.strategy == "sentence":
                masked = 0
    
        return _X

    def get_masked_min_len_max_count(self, X : List[List[str]], ratio : float) -> Tuple[int, int]:
        if any(isinstance(x, list) for x in X):
            # X is a List[List[str]], need to flatten it
            X = [t for x in X for t in x]
        
        max_masked_count = int(len(X) * ratio)
        min_masked_len = max(1, sorted([len(t) for t in X], reverse=True)[max_masked_count-1])

        return min_masked_len, max_masked_count


class FrequencyBasedMask(Mask):
    def __init__(self, top_freq, mask_token) -> None:
        assert(top_freq > 0), "Top frequent must be positive"
        self.top_freq = top_freq
        self.mask_token = mask_token
    
    def mask(self, X: List[List[str]]) -> List[List[str]]:
        """X is a list of lists of strings, a string represents one token"""
        # make a copy of the data not to mess up the orogonal
        _X = copy.deepcopy(X)

        tokens_to_mask = self.get_most_frequent(X=_X, top_freq=self.top_freq)

        for i in range(len(_X)):
            for j in range(len(i)):
                if _X[i][j] in tokens_to_mask:
                    _X[i][j] = self.mask_token
        return _X
    
    def get_most_frequent(X, top_freq) -> List[str]:
        # flatten data
        X = [t for x in X for t in x]
        most_frequent = Counter(X).most_common(top_freq)
        tokens_to_mask = [x[0] for x in most_frequent]

        return tokens_to_mask


class ProbabilisticMask(Mask):
    def __init__(self, mask_probas, mask_token) -> None:
        self.mask_probas = mask_probas
        self.mask_token = mask_token
    