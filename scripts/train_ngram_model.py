import json
import os

import pickle

from nltk.lm import MLE, KneserNeyInterpolated, Lidstone
from sklearn.model_selection import ParameterGrid
from src.models.masking import (
    FrequencyMask,
    LengthWindowMask,
    POSMask,
    RandomWindowMask,
    StopwordMask,
    WeightedFrequencyMask,
)
from src.models.ngram_model import NGramModel
from src.utils.eval_metrics import harmonic_mean
from src.data.data_loader import load_data


def main():
    # Create grid of parameters to try
    lm_grid = [
        {"lm": [MLE], "n": [1, 2, 3]},
        {"lm": [Lidstone], "n": [1, 2, 3, 4, 5], "gamma": [0.1, 0.5, 0.7, 1, 2, 3]},
        {
            "lm": [KneserNeyInterpolated],
            "n": [2, 3, 4, 5],
            "discount": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
    ]
    lm_grid_list = list(ParameterGrid(lm_grid))
    # Get train dataset
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    TRAINING_DATA_PATH = os.path.join(cur_dir, "..", "data", "clean", "training")
    X_train = load_data(TRAINING_DATA_PATH)
    # Load validation dataset
    VALIDATION_DATA_PATH = os.path.join(cur_dir, "..", "data", "clean", "validation")
    X_val = load_data(VALIDATION_DATA_PATH)
    # Load the masked dataset
    MASKED_DATA_PATH = os.path.join(cur_dir, "..", "src", "models", "pickle", "masking", "maskig")

    # Train the language models
    evaluation_dict = {}
    for i, params in enumerate(lm_grid_list[:1]):
        print(f"Training n-gram language model with id = {i} with params : \n{params}")
        pickle_path = os.path.join(cur_dir, "..", "src", "models", "pickle", "ngram", f"lm_{i}.pkl")
        model = NGramModel(n=params["n"])
        kwargs = {k: v for k, v in params.items() if k != "lm" and k != "n"}
        dict_to_log = {"id": i, "lm": params["lm"].__name__, "n": params["n"], **kwargs}
        if os.path.exists(pickle_path):
            model.load_model(pickle_path)
        else:
            model.fit(X_train, lm=params["lm"], **kwargs)
            model.save_model(
                pickle_model_path=os.path.join(
                    cur_dir, "..", "src", "models", "pickle", "ngram", f"lm_{i}.pkl"
                )
            )
        # Evaluate the model on each masking dataset
        for key, X_masked in X_masked_dict.items():
            # This appears to be a time bottleneck
            import time
            start = time.time()
            X_decoded = model.decode(X_masked[:5], parallel=True)
            print(time.time() - start)
            print(X_val[:5])
            print(X_masked[:5])
            print(X_decoded[:5])
            reconstruction_accuracy = model.accuracy_score(
                masking_token=MASK_TOKEN,
                X_original=X_val[:5],
                X_masked=X_masked[:5],
                X_decoded=X_decoded[:5],
            )
            similarity_score = model.similarity_score(
                masking_token=MASK_TOKEN,
                X_original=X_val[:5],
                X_masked=X_masked[:5],
                X_decoded=X_decoded[:5],
            )
            evaluation_dict[(f"lm_{i}", *key)] = {
                "reconstruction_accuracy": reconstruction_accuracy,
                "similarity_score": similarity_score,
                "compression_score": compression_score[key],
                "harmonic_accuracy_compression": harmonic_mean(
                    [reconstruction_accuracy, compression_score[key]]
                ),
                "harmonic_similarity_compression": harmonic_mean(
                    [similarity_score, compression_score[key]]
                ),
            }
            print(evaluation_dict)


if __name__ == "__main__":
    main()
