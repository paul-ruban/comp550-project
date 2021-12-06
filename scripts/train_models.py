import argparse
import json
import logging
import os
import pickle
import re

import numpy as np
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from src.utils.json_utils import append_json_lines, read_json_lines

cur_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger()

# Classes and constants #

LOGGER_FOLDER_PATH = os.path.join(cur_dir, "..", "logs", "training")
DATA_TYPE_DICT = {
    "polarity": {
        "json_training_log_path": os.path.join(
            cur_dir, "..", "logs", "augmentation", "polarity.json"
        ),
        "json_validation_path": os.path.join(
            cur_dir, "..", "data", "rt-polaritydata", "augmentation", "validation.json"
        ),
        "json_test_path": os.path.join(
            cur_dir, "..", "data", "rt-polaritydata", "augmentation", "test.json"
        ),
        "json_train_log_path": os.path.join(cur_dir, "..", "logs", "training", "polarity.json"),
        "pickle_folder_path": os.path.join(cur_dir, "..", "temp", "training", "polarity"),
    },
    "articles": {
        "json_training_log_path": os.path.join(
            cur_dir, "..", "logs", "augmentation", "articles.json"
        ),
        "json_validation_path": os.path.join(
            cur_dir, "..", "data", "articles", "augmentation", "validation.json"
        ),
        "json_test_path": os.path.join(
            cur_dir, "..", "data", "articles", "augmentation", "test.json"
        ),
        "json_train_log_path": os.path.join(cur_dir, "..", "logs", "training", "articles.json"),
        "pickle_folder_path": os.path.join(cur_dir, "..", "temp", "training", "polarity"),
    },
    "smokers": {
        "json_training_log_path": os.path.join(
            cur_dir, "..", "logs", "augmentation", "smokers.json"
        ),
        "json_validation_path": [
            os.path.join(
                cur_dir, "..", "data", "smokers", "augmentation", f"validation_fold_{i+1}.json"
            )
            for i in range(1, 6)
        ],
        "json_test_path": os.path.join(
            cur_dir, "..", "data", "smokers", "augmentation", "test.json"
        ),
        "json_train_log_path": os.path.join(cur_dir, "..", "logs", "training", "smokers.json"),
        "pickle_folder_path": os.path.join(cur_dir, "..", "temp", "training", "polarity"),
    },
}
MODELS = {"logistic", "svm", "mlp"}


class WordTokenizer:
    def __init__(self, remove_punctuation=True):
        self.remove_punctuation = remove_punctuation

    def __call__(self, doc):
        if not self.remove_punctuation:
            return [t for t in word_tokenize(doc)]
        else:
            return [
                t for t in word_tokenize(doc) if not re.search(r"^[\.,\?\!\:\;\(\)\[\]\{\}]*$", t)
            ]


class LemmaTokenizer:
    def __init__(self, remove_punctuation=True):
        self.wnl = WordNetLemmatizer()
        self.remove_punctuation = remove_punctuation

    def __call__(self, doc):
        if not self.remove_punctuation:
            return [self.wnl.stem(t) for t in word_tokenize(doc)]
        else:
            return [
                self.wnl.lemmatize(t)
                for t in word_tokenize(doc)
                if not re.search(r"^[\.,\?\!\:\;\(\)\[\]\{\}]*$", t)
            ]


class StemmerTokenizer:
    def __init__(self, remove_punctuation=True):
        self.wnl = PorterStemmer()
        self.remove_punctuation = remove_punctuation

    def __call__(self, doc):
        if not self.remove_punctuation:
            return [self.wnl.stem(t) for t in word_tokenize(doc)]
        else:
            return [
                self.wnl.stem(t)
                for t in word_tokenize(doc)
                if not re.search(r"^[\.,\?\!\:\;\(\)\[\]\{\}]*$", t)
            ]


# Helper functions #


def parse_args():
    parser = argparse.ArgumentParser("Trains given the data augmentation for the passed data type")
    parser.add_argument(
        "-t", "--data_type", help=f"Data type must be one of {DATA_TYPE_DICT.keys()}"
    )
    args = parser.parse_args()
    data_type = args.data_type
    assert data_type in DATA_TYPE_DICT.keys()
    return data_type


def prepare_train_and_val_data(training_json_path: str, validation_json_path: str):
    # Load the training and validation data
    training_data = read_json_lines(training_json_path)
    validation_data = read_json_lines(validation_json_path)
    # Convert to numpy arrays
    training_matrix = np.array([[dict_["text"], dict_["label"]] for dict_ in training_data])
    validation_matrix = np.array([[dict_["text"], dict_["label"]] for dict_ in validation_data])
    # Split into X and y
    X_train, y_train = training_matrix[:, 0], training_matrix[:, 1]
    X_val, y_val = validation_matrix[:, 0], validation_matrix[:, 1]
    # To use GridSearch format merge train and val and keep the split index
    # By convention training points are identified with -1 and validation points with 0
    split_index = [-1] * len(X_train) + [0] * len(X_val)
    X = np.vstack((X_train, X_val))
    y = np.hstack((y_train, y_val))
    return X, y, split_index


def get_test_data(test_json_path: str):
    test_data = read_json_lines(test_json_path)
    test_matrix = np.array([[dict_["text"], dict_["label"]] for dict_ in test_data])
    X_test, y_test = test_matrix[:, 0], test_matrix[:, 1]
    return X_test, y_test


def create_pipeline(
    model_type: str = "logistic",
) -> Pipeline:
    # Use the "meta" parameters to create the full sklearn pipeline, then use GridSearchCV
    # to find the best paramters (allows multithreading)
    assert model_type in {"logistic", "svm", "mlp"}

    # Pipeline will be a pipeline object with a counter and a model
    pipeline = []
    # Use tfidf vectorizer
    vectorizer = TfidfVectorizer()
    pipeline.append(("vectorizer", vectorizer))
    # Pick the model
    if model_type == "logistic":
        model = LogisticRegression()
    elif model_type == "svm":
        model = SVC()
    elif model_type == "mlp":
        model = MLPClassifier()
    pipeline.append(("model", model))
    pipeline = Pipeline(pipeline)
    return pipeline


def create_pipeline_grid(model_type: str = "logistic"):
    assert model_type in {"logistic", "svm", "mlp"}
    param_grid = {
        # Punctuation removal is on automatically
        "vectorizer__tokenizer": [WordTokenizer(), LemmaTokenizer(), StemmerTokenizer()],
        "vectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)],
    }
    if model_type == "logistic":
        model_grid = {
            "model__penalty": ["l2"],
            "model__C": [0.01, 0.1, 1],
            "model__solver": ["lbfgs"],
            "model__random_state": [42],
        }
    elif model_type == "svm":
        model_grid = {
            "model__C": [0.01, 0.1, 1],
            "model__kernel": ["poly", "rbf"],
            "model__random_state": [42],
        }
    elif model_type == "mlp":
        model_grid = {
            "model__hidden_layer_sizes": [(100,), (100, 100), (100, 100, 100)],
            "model__activation": ["relu"],
            "model__solver": ["adam"],
            "model__random_state": [42],
            "model__tol": [1e-4],
        }
    pipeline_grid = {**param_grid, **model_grid}
    return pipeline_grid


def train_pipeline(
    X: np.ndarray, y: np.ndarray, split_index: np.ndarray, pieline: Pipeline, grid: dict
):
    # Use GridSearchCV to find the best paramters (allows multithreading)
    pds = PredefinedSplit(test_fold=split_index)
    grid_search = GridSearchCV(
        estimator=pieline, param_grid=grid, cv=pds, scoring="f1", n_jobs=-1, verbosity=3
    )
    grid_search.fit(X, y, split_index)
    return grid_search


def log_grid_search_results(grid_search: GridSearchCV, X_test: np.ndarray, y_test: np.ndarray):
    # Log the results of the grid search
    logger.info("Results of the grid search for the model:")
    logger.info("Grid search results:")
    logger.info(f"{json.dumps(grid_search.cv_results_, indent=4)}")
    logger.info("Best estimator:")
    logger.info(f"{grid_search.best_estimator_}")
    logger.info("Best parameters:")
    logger.info(f"{grid_search.best_params_}")
    logger.info("Best (f1) score:")
    logger.info(f"{grid_search.best_score_}")
    # Evaluate on the test set
    y_pred = grid_search.best_estimator_.predict(X_test)
    logger.info("Test set results of the best classifier:")
    logger.info(f"{classification_report(y_true=y_test, y_pred=y_pred)}")


def save_grid_search_results(
    augmentation_id: int,
    augmentation_features: dict,
    model_type: str,
    grid_search: GridSearchCV,
    pickle_folder_path: str,
    json_file_path: str,
):
    # Pickle the grid search object
    grid_search_pickle_path = os.path.join(
        pickle_folder_path, f"grid_search_augmentation_{augmentation_id}_{model_type}.pkl"
    )
    with open(grid_search_pickle_path, "wb") as f:
        pickle.dump(grid_search, f)
    # Append the results of the grid search to the json file
    dict_to_write = [
        {
            "augmentation_dataset_id": augmentation_id,
            "augmentation_features": augmentation_features,
            "model_type": model_type,
            "score": grid_search.best_score_,
            "path_to_pickle": grid_search_pickle_path,
        }
    ]
    append_json_lines(dict_to_write, json_file_path)


def main():
    # Get the data type
    data_type = parse_args()
    # Setup the logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOGGER_FOLDER_PATH, f"{data_type}.json")),
            logging.StreamHandler(),
        ],
    )
    logger.info(f"Starting {data_type} training")
    # Get the json file for the augmentation
    augmentation_json_file_path = DATA_TYPE_DICT[data_type]["json_training_log_path"]
    # Get each dictionary in the json file
    augmentation_dicts = read_json_lines(augmentation_json_file_path)
    for augmentation_dict in augmentation_dicts:
        logger.info("-" * 60)
        logger.info(
            f"Starting training for {data_type} with augmentation {augmentation_dict['augmentation_dataset_id']}"
        )
        logger.info(f"Augmentation features: {augmentation_dict['augmentation_features']}")
        # Get train, val, test paths
        training_json_path = augmentation_dict["path_to_augmented_dataset"]
        validation_json_path = DATA_TYPE_DICT[data_type]["json_validation_path"]
        test_json_path = DATA_TYPE_DICT[data_type]["json_test_path"]
        # Get data for pipeline and testing data
        X, y, split_index = prepare_train_and_val_data(
            training_json_path=training_json_path, validation_json_path=validation_json_path
        )
        X_test, y_test = get_test_data(test_json_path=test_json_path)
        # Train the different models
        for model_type in MODELS:
            logger.info("-" * 60)
            logger.info(f"Starting training for {model_type}")
            # Create the pipeline
            pipeline = create_pipeline(model_type=model_type)
            # Create the grid
            grid = create_pipeline_grid(model_type=model_type)
            # Train the pipeline
            grid_search = train_pipeline(
                X=X, y=y, split_index=split_index, pieline=pipeline, grid=grid
            )
            # Log the grid search results on the validation set and the test set
            log_grid_search_results(grid_search=grid_search, X_test=X_test, y_test=y_test)
            # Save the results
            save_grid_search_results(
                augmentation_id=augmentation_dict["augmentation_dataset_id"],
                augmentation_features=augmentation_dict["augmentation_features"],
                model_type=model_type,
                grid_search=grid_search,
                pickle_folder_path=DATA_TYPE_DICT[data_type]["pickle_folder_path"],
                json_file_path=DATA_TYPE_DICT[data_type]["json_train_log_path"],
            )


if __name__ == "__main__":
    main()
