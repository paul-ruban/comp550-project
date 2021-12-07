import os
from typing import Tuple

import random
import nlpaug
import nlpaug.augmenter.word as naw
import numpy as np
import torch
from src.utils.json_utils import append_json_lines, write_json_lines
from pathlib import Path


class Augmentation:
    AUGMENTATION_TYPES = [
        "none",
        "random_swap",
        "random_delete",
        "synonym_wordnet",
        "synonym_word2vec",
        "backtranslation",
        "ssmba",
    ]

    def __init__(
        self, augmentation_type: str, num_samples: int = 1, random_state: int = 42
    ) -> None:
        """Augmentation module.

        Args:
            augmentation_type (str): Type of
            num_samples (int): Number of NEW samples to generate per ORIGINAL samples.
        """
        assert (
            augmentation_type in Augmentation.AUGMENTATION_TYPES
        ), f"The augmentation type must be one of {Augmentation.AUGMENTATION_TYPES}"
        self.augmentation_type = augmentation_type
        self.num_samples = num_samples
        self.random_state = random_state

    def augment(self, X: np.array, y: np.array, **kwargs) -> Tuple[np.array, np.array]:
        """Main augmentation function. Takes D = {(x_i, y_i) : i = 1 to N} and
        creates an augmented dataset D_aug = {(x_ik, y_ik) : k = 1 to self.num_sample, i = 1 to N}

        **IMPORTANT NOTE: This assumes that the order of D is preserved in the order of D_aug. You
        must ensure that this is true for the saving. (If, for whatever reason, a sample i does not
        have self.num_samples samples generated, then an empty string should be added.)


        **IMPORTANT NOTE: The nlpaug and ssmba use tokenizers before applying augmentation.
        For nlpaug the tokenizer (I think) is the standard space tokenizer, so make sure you
        have space separated everything. For ssmba they use the model's tokenizer.

        Args:
            X (np.array): Input text array, expected to be one dimensional
            y (np.array): Input label array, expected to be one dimensional
            kwargs: List of keyword arguments to pass to the data augmenter

        returns : The augmented input and label
        """
        assert (
            len(X.shape) == len(y.shape) == 1
        ), "Both X and y should have dimension 1."
        assert X.size == y.size, "Both X and y should have the same size."
        # set kwargs for the logger
        self.kwargs = kwargs
        # cast X to object type
        X = X.astype(dtype="object") if X.dtype != np.dtype("object") else X
        if self.augmentation_type == "none":
            return np.array([]), np.array([])
        elif self.augmentation_type == "random_swap":
            return self.random_swap_augment(X, y, **kwargs)
        elif self.augmentation_type == "random_delete":
            return self.random_delete_augment(X, y, **kwargs)
        elif self.augmentation_type == "synonym_wordnet":
            return self.synonym_wordnet(X, y, **kwargs)
        elif self.augmentation_type == "synonym_word2vec":
            return self.synonym_word2vec(X, y, **kwargs)
        elif self.augmentation_type == "backtranslation":
            return self.backtranslation(X, y, **kwargs)
        elif self.augmentation_type == "ssmba":
            return self.ssmba(X, y, **kwargs)
        else:
            raise ValueError("Augmentation type not supported")

    def to_json(
        self,
        path_to_folder: str,
        X_initial: np.array,
        y_initial: np.array,
        X_aug: np.array,
        y_aug: np.array,
        name: str = "training",
    ) -> str:
        """Method that will write the augmented dataset to json so that it has the same format as val and
        test.json. Should be saved as train_id.json and will have the following format:
        1.  {"id": 1, "is_original": True, "label": 1, "text": "This movie sucked"}
        2.  {"id": 1, "is_original": False, "label": 1, "text": "This movie blowed"}

        Args:
            id (str): Name to identify the saved training set (the user should specify this in an outside training loop)
            path_to_folder (str): Path to folder to store json folder
            X_initial (np.array): Initial non-augmented input array
            y_initial (np.array): Initial no-augmented label array
            X_aug (np.array): Augmented input array
            y_aug (np.array): Augmented label array
            kwargs: Additional keyword arguments. May be necessary for the clinical text.

        returns : The path to the json (which is also recorded as an attribute)
        """
        assert X_aug.ndim == y_aug.ndim == X_initial.ndim == y_initial.ndim == 1
        assert X_initial.size == y_initial.size
        assert X_aug.size == y_aug.size
        assert X_aug.size == (self.num_samples * X_initial.size)
        if not os.path.exists(path_to_folder):
            print(f"{path_to_folder} doesn't exist, making it.")
            os.makedirs(path_to_folder)
        list_of_dict_to_write = []
        # Might need to do something different here for the medical texts
        for i, (x_i, y_i) in enumerate(zip(X_initial, y_initial)):
            # Place an int() wrapper to suppress the json error
            dict_to_write = {
                "id": i,
                "is_original": True,
                "label": int(y_i),
                "text": x_i,
            }
            list_of_dict_to_write.append(dict_to_write)
            # So it looks like the formula
            k = self.num_samples
            range_ = slice(i * k, (i + 1) * k)
            for x_ik, y_ik in zip(X_aug[range_], y_aug[range_]):
                dict_to_write = {
                    "id": i,
                    "is_original": False,
                    "label": int(y_ik),
                    "text": x_ik,
                }
                list_of_dict_to_write.append(dict_to_write)
        # Actually do the writing! Set this as an attribute to use for the logger
        self.path_to_json = os.path.join(path_to_folder, f"{name}.json")
        write_json_lines(
            list_to_write=list_of_dict_to_write, output_path=self.path_to_json
        )
        return self.path_to_json

    def log_to_json(self, id: int, time_now: str, path_to_json_log: str) -> None:
        """Method that will record the id, augmentation features and path to the augmented dataset into
        a json file. The point of this is so that training on the different datasets can be seamless.

        The format should be:
        {
            "augmented_dataset_id": id,
            "time_now": time_now,
            "augmentation_features": {
                "num_sample": self.number_samples,
                "augmentation_type": self.augmentation_type
                **self.kwargs
            },
            "path_to_augmented_dataset": path
        }

        Args:
            id (int): Id to identify the training set (this should be the same as to_json)
            path_to_json_log (str): Path to the json logger. Creates it if doesn't already exist.
        """
        if self.path_to_json is None:
            raise ValueError(
                "Cannot log the augmentation if the dataset hasn't been saved!"
            )
        path_to_json_log = Path(path_to_json_log)
        if not path_to_json_log.parent.exists():
            print(f"{path_to_json_log.parent} folder does not exist, creating it")
            os.makedirs(path_to_json_log.parent)
        dict_to_write = {
            "augmented_dataset_id": id,
            "time_now": time_now,
            "augmentation_features": {
                "num_samples": self.num_samples,
                "augmentation_type": self.augmentation_type,
                **self.kwargs,
            },
            "path_to_augmented_dataset": self.path_to_json,
        }
        append_json_lines([dict_to_write], output_path=path_to_json_log)

    def _augment(self, X: np.array, y: np.array, aug: nlpaug.augmenter.word):
        random.seed(self.random_state)
        X_aug = []
        for x in X:
            if self.num_samples == 1:
                X_aug += [
                    aug.augment(x, n=self.num_samples, num_thread=self.num_samples)
                ]
            else:
                X_aug += aug.augment(x, n=self.num_samples, num_thread=self.num_samples)
        X_aug = np.array(X_aug)
        y_aug = np.repeat(y, repeats=self.num_samples)
        return X_aug, y_aug

    def random_swap_augment(self, X: np.array, y: np.array, aug_p: float, **kwargs):
        aug = naw.RandomWordAug(action="swap", aug_p=aug_p, **kwargs)
        return self._augment(X, y, aug)

    def random_delete_augment(self, X: np.array, y: np.array, aug_p: float, **kwargs):
        aug = naw.RandomWordAug(action="delete", aug_p=aug_p, **kwargs)
        return self._augment(X, y, aug)

    def synonym_wordnet(self, X: np.array, y: np.array, aug_p: float, **kwargs):
        aug = naw.SynonymAug(aug_src="wordnet", aug_p=aug_p, **kwargs)
        return self._augment(X, y, aug)

    def synonym_word2vec(
        self, X: np.array, y: np.array, aug_p: float, top_k: int, **kwargs
    ):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        word2vec_bin_path = os.path.join(
            cur_dir, "..", "..", "GoogleNews-vectors-negative300.bin"
        )
        assert os.path.exists(
            word2vec_bin_path
        ), "You must download and unzip the w2vec file and place it in the root directory of the repo."
        aug = naw.WordEmbsAug(
            model_type="word2vec",
            model_path=word2vec_bin_path,
            action="substitute",
            aug_p=aug_p,
            top_k=top_k,
            **kwargs,
        )
        return self._augment(X, y, aug)

    def backtranslation(self, X: np.array, y: np.array):
        # Use the en -> de -> en translation
        # Cut the input array to the max length
        MAX_LEN = 1024
        X_truncated = [x[:MAX_LEN] for x in X]
        aug = naw.BackTranslationAug(
            from_model_name="facebook/wmt19-en-de",
            to_model_name="facebook/wmt19-de-en",
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_length=1024,  # This is the model's max length
        )
        return self._augment(X_truncated, y, aug)

    def ssmba(self, X, y):
        """TODO: Modify the paper's code slightly to make this work. Not using nlpaug because
        we can test a few relaxing assumptions such as temperature and the use of a more closed/restrcited
        space for augmentation.
        """
        pass
