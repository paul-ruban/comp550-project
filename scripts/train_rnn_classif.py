import argparse
import logging
import os
from functools import partial
import contextlib

import torch
from torch import nn
from src.models.rnn_classif import JsonDataset, RNNClassifier, RNNTrainer
from src.utils.json_utils import append_json_lines, read_json_lines
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import ParameterGrid

cur_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger()


# CONSTANTS #

LOGGER_FOLDER_PATH = os.path.join(cur_dir, "..", "logs", "training_rnn_classif")
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
        "json_train_log_path": os.path.join(
            cur_dir, "..", "logs", "training_rnn_classif", "polarity.json"
        ),
        # Change to save to the augmentation
        "pickle_folder_path": "/home/mila/c/cesare.spinoso/scratch/datasets_550/polarity",
        # "pickle_folder_path": "/home/c_spino/comp_550/comp-550-project/data/temp/polarity",
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
        "json_train_log_path": os.path.join(
            cur_dir, "..", "logs", "training_rnn_classif", "articles.json"
        ),
        # Change to save to the augmentation
        "pickle_folder_path": "/home/mila/c/cesare.spinoso/scratch/datasets_550/articles",
    },
    "smokers": {
        "json_training_log_path": os.path.join(
            cur_dir, "..", "logs", "augmentation", "smokers.json"
        ),
        "json_validation_path": os.path.join(
            cur_dir, "..", "data", "smokers", "augmentation", "validation.json"
        ),
        "json_test_path": os.path.join(
            cur_dir, "..", "data", "smokers", "augmentation", "test.json"
        ),
        "json_train_log_path": os.path.join(
            cur_dir, "..", "logs", "training_rnn_classif", "smokers.json"
        ),
        # Change to save to the augmentation
        "pickle_folder_path": "/home/mila/c/cesare.spinoso/scratch/datasets_550/smokers",
    },
}

MODELS = {
    "polarity": [
        "distilbert-base-uncased",
    ],
    "articles": ["distilbert-base-uncased"],
    "smokers": [
        "distilbert-base-uncased",
        "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    ],
}

MODEL_FILE_NAME = {
    "distilbert-base-uncased": "distilbert_base_uncased",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext": "pubmed_bert_base_uncased",
}

OPTIMIZER = torch.optim.Adam
LOSS = torch.nn.CrossEntropyLoss()

HYPERPARAMETER_GRID = {
    "model_type": ["lstm"],
    "lr": [0.001],
    "num_epochs": [100],
    "batch_size": [32],
    "hidden_dim": [256],
    "num_layers": [1],
    "dropout": [0.2],
    # "bidirectional": [True, False],
    "bidirectional": [True,],
}

OUTPUT_DIM = {
    "polarity": 2,
    "articles": 5,
    "smokers": 3,
}

LOG_INTERVAL = {
    "polarity": 50,
    "articles": 10,
    "smokers": 5
}

# Functions #


def parse_args():
    parser = argparse.ArgumentParser("Trains given the data augmentation for the passed data type")
    parser.add_argument(
        "-t", "--data_type", help=f"Data type must be one of {DATA_TYPE_DICT.keys()}"
    )
    args = parser.parse_args()
    data_type = args.data_type
    assert data_type in DATA_TYPE_DICT.keys()
    return data_type


def setup_logger(data_type):
    if not os.path.exists(LOGGER_FOLDER_PATH):
        os.makedirs(LOGGER_FOLDER_PATH)
    # Setup the logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOGGER_FOLDER_PATH, f"{data_type}.log")),
            logging.StreamHandler(),
        ],
    )
    # NOTE: If you are not seeing a debug message it maybe be because of this
    logging.disable(logging.DEBUG)


def get_augmentation_dict(data_type):
    # Get the augmentation json file path
    augmentation_json_file_path = DATA_TYPE_DICT[data_type]["json_training_log_path"]
    # Get each dictionary in the json file
    augmentation_dicts = read_json_lines(augmentation_json_file_path)
    return augmentation_dicts


def train_models(data_type, augmentation_dicts):
    for augmentation_dict in augmentation_dicts:
        if data_type == "polarity":
            if augmentation_dict['augmented_dataset_id'] not in {1, 2, 3}:
                continue
        if data_type == "articles":
            if augmentation_dict['augmented_dataset_id'] not in {5, 6, 9, 11, 12}:
                continue
        logger.info("+" * 90)
        logger.info(
            f"Starting training for {data_type} with augmentation {augmentation_dict['augmented_dataset_id']}"
        )
        logger.info(f"Augmentation features: {augmentation_dict['augmentation_features']}")
        for model_type in MODELS[data_type]:
            logger.info("~" * 75)
            logger.info(f"Starting training using the HuggingFace model {model_type}")
            # Get the model and tokenizer from huggingface
            model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_type)
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_type)
            # Get train, val, test paths
            training_json_path = augmentation_dict["path_to_augmented_dataset"]
            validation_json_path = DATA_TYPE_DICT[data_type]["json_validation_path"]
            test_json_path = DATA_TYPE_DICT[data_type]["json_test_path"]
            # Load the datasets
            train_dataset = JsonDataset(training_json_path, tokenizer, data_type)
            val_dataset = JsonDataset(validation_json_path, tokenizer, data_type)
            test_dataset = JsonDataset(test_json_path, tokenizer, data_type)
            logger.info("Done loading all the datasets")
            # Create the search grid
            hyperparam_grid = ParameterGrid(HYPERPARAMETER_GRID)
            # Running parameter to get the best model from the hyperparameters
            best_rnn_state_dict = {}
            best_f1_score = 0.0
            best_accuracy = 0.0
            best_hyperparam = {}
            for hyperparam in hyperparam_grid:
                logger.info("=" * 60)
                logger.info(f"Starting training with the following hyperparameters: {hyperparam}")
                # Create the data loaders
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=hyperparam["batch_size"],
                    shuffle=True,
                    collate_fn=partial(train_dataset.collate_batch, tokenizer=tokenizer),
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=hyperparam["batch_size"],
                    shuffle=True,
                    collate_fn=partial(train_dataset.collate_batch, tokenizer=tokenizer),
                )
                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=hyperparam["batch_size"],
                    shuffle=True,
                    collate_fn=partial(train_dataset.collate_batch, tokenizer=tokenizer),
                )
                # Set up the model and its loss + optimizer
                with contextlib.redirect_stdout(None):
                    rnn = RNNClassifier(
                        model_type=hyperparam["model_type"],
                        pretrained_word_embedding=model.embeddings.word_embeddings,
                        hidden_dim=hyperparam["hidden_dim"],
                        num_layers=hyperparam["num_layers"],
                        output_dim=OUTPUT_DIM[data_type],
                        bidirectional=hyperparam["bidirectional"],
                        dropout=hyperparam["dropout"],
                    )
                optimizer = torch.optim.Adam(rnn.parameters(), lr=hyperparam["lr"])
                loss = nn.CrossEntropyLoss()
                # Train the model
                rnn_trainer = RNNTrainer(
                    model=rnn,
                    optimizer=optimizer,
                    criterion=loss,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    logger=logger,
                    num_epochs=hyperparam["num_epochs"],
                    log_interval=LOG_INTERVAL[data_type]
                )
                rnn_trainer.train()
                if rnn_trainer.best_f1_score > best_f1_score:
                    best_f1_score = rnn_trainer.best_f1_score
                    best_accuracy = rnn_trainer.best_valid_acc
                    best_rnn_state_dict = {
                        "model": rnn_trainer.best_model,
                        "optimizer": rnn_trainer.best_optimizer,
                        "loss": rnn_trainer.best_loss,
                    }
                    best_hyperparam = hyperparam
            # Log val and test results
            logger.info("=" * 60)
            logger.info(f"Finished training model. Best hyperparameters: {best_hyperparam}")
            logger.info("Here is the validation results:")
            RNNTrainer.report_metrics(
                model=best_rnn_state_dict["model"], dataloader=val_dataloader, logger=logger
            )
            logger.info(
                "Here is the test results (DO NOT USE THESE RESULTS FOR CHOOSING THE BEST AUGMENTATION):"
            )
            RNNTrainer.report_metrics(
                model=best_rnn_state_dict["model"], dataloader=test_dataloader, logger=logger
            )
            # Save the best model
            model_save_folder = os.path.join(
                DATA_TYPE_DICT[data_type]["pickle_folder_path"],
                f"augmentation_{augmentation_dict['augmented_dataset_id']}",
            )
            model_path = os.path.join(
                model_save_folder,
                f"rnn_classif_{MODEL_FILE_NAME[model_type]}.pt",
            )
            if not os.path.exists(model_save_folder):
                os.makedirs(model_save_folder)
            RNNTrainer.save_checkpoint(
                save_path=model_path,
                model=best_rnn_state_dict["model"],
                optimizer=best_rnn_state_dict["optimizer"],
                loss=best_rnn_state_dict["loss"],
            )
            # Log this to the json
            json_dict = {
                "augmented_dataset_id": augmentation_dict["augmented_dataset_id"],
                "augmentation_features": augmentation_dict["augmentation_features"],
                "huggingface_model": model_type,
                "f1_score": best_f1_score,
                "accuracy_score": best_accuracy,
                "path_to_pickle": model_path,
            }
            append_json_lines(
                [json_dict], output_path=DATA_TYPE_DICT[data_type]["json_train_log_path"]
            )


def main():
    # Get the data type
    data_type = parse_args()
    # Setup the logger
    setup_logger(data_type)
    # Get augmentation dicts
    augmentation_dicts = get_augmentation_dict(data_type)
    # Train models for each augmentation dict
    # NOTE: The training is non-deterministic to speed up computation
    # This is why the model is saved and the TEST eval is measured at the same time
    logger.info("-" * 120)
    logger.info(f"Training models for {data_type}")
    train_models(data_type, augmentation_dicts)


if __name__ == "__main__":
    main()
