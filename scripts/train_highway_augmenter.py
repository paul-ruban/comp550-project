import argparse
import logging
import os
from copy import deepcopy
import contextlib

import torch
from torch import nn

from src.data.dataio import Dataset
from src.models.rnn_model import RNN
from src.models.learned_masking import HighwayAugmenter, HighwayAugmenterTrainer, WeightedMaskClassificationLoss
from src.utils.json_utils import append_json_lines
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import ParameterGrid

cur_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger()


# CONSTANTS #

LOGGER_FOLDER_PATH = os.path.join(cur_dir, "..", "logs", "training_rnn_classif")
DATA_TYPE_DICT = {
    "polarity": {
        "json_train_path": os.path.join(
            cur_dir, "..", "data", "rt-polaritydata", "augmentation", "train.json"
        ),
        "json_validation_path": os.path.join(
            cur_dir, "..", "data", "rt-polaritydata", "augmentation", "validation.json"
        ),
        "json_test_path": os.path.join(
            cur_dir, "..", "data", "rt-polaritydata", "augmentation", "test.json"
        ),
        "json_train_log_path": os.path.join(
            cur_dir, "..", "logs", "training_highway_augmenter", "polarity.json"
        ),
        # Change to save to the augmentation
        # "pickle_folder_path": "/home/mila/c/cesare.spinoso/scratch/datasets_550/polarity",
        "pickle_folder_path": "/home/c_spino/comp_550/comp-550-project/data/temp/polarity",
    },
    "articles": {
        "json_train_path": os.path.join(
            cur_dir, "..", "data", "articles", "augmentation", "train.json"
        ),
        "json_validation_path": os.path.join(
            cur_dir, "..", "data", "articles", "augmentation", "validation.json"
        ),
        "json_test_path": os.path.join(
            cur_dir, "..", "data", "articles", "augmentation", "test.json"
        ),
        "json_train_log_path": os.path.join(
            cur_dir, "..", "logs", "training_highway_augmenter", "articles.json"
        ),
        # Change to save to the augmentation
        "pickle_folder_path": "/home/mila/c/cesare.spinoso/scratch/datasets_550/articles",
    },
    "smokers": {
        "json_train_path": os.path.join(
            cur_dir, "..", "data", "smokers", "augmentation", "train.json"
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
    "polarity": ["distilbert-base-uncased"],
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
LOSS = WeightedMaskClassificationLoss()

HYPERPARAMETER_GRID = {
    "model_type": ["lstm"],
    "lr": [0.001],
    "num_epochs": [50],
    "batch_size": [32],
    "hidden_dim": [256],
    "num_layers": [1],
    "dropout": [0.2],
    "bidirectional": [True, False],
    "max_seq_length": [64]
}

OUTPUT_DIM = {
    "polarity": 2,
    "articles": 5,
    "smokers": 3,
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


def train_models(data_type):
    logger.info("+" * 90)
    logger.info(
        f"Starting training for {data_type} with learning augmentation."
    )
    for model_type in MODELS[data_type]:
        logger.info("~" * 75)
        logger.info(f"Starting training using the HuggingFace model {model_type}")
        # Get the model and tokenizer from huggingface
        bert_model = AutoModel.from_pretrained(model_type)
        bert_tokenizer = AutoTokenizer.from_pretrained(model_type)
        # Get train, val, test paths
        training_json_path = DATA_TYPE_DICT[data_type]["json_train_path"]
        validation_json_path = DATA_TYPE_DICT[data_type]["json_validation_path"]
        test_json_path = DATA_TYPE_DICT[data_type]["json_test_path"]
        # Load the datasets
        train_dataset = Dataset([training_json_path], data_format="json")
        val_dataset = Dataset(validation_json_path, data_format="json")
        test_dataset = Dataset(test_json_path, data_format="json")
        logger.info("Done loading all the datasets")
        # Create the search grid
        hyperparam_grid = ParameterGrid(HYPERPARAMETER_GRID)
        # Running parameter to get the best model from the hyperparameters
        best_state_dict = {}
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
                shuffle=True
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=hyperparam["batch_size"],
                shuffle=True
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=hyperparam["batch_size"],
                shuffle=True
            )
            # Set up the model and its loss + optimizer
            with contextlib.redirect_stdout(None):
                model = HighwayAugmenter(
                    tokenizer=bert_tokenizer,
                    masking_model=RNN(
                        rnn_type=hyperparam["model_type"],
                        embeddings_layer=deepcopy(bert_model.embeddings.word_embeddings),
                        hidden_dim=hyperparam["hidden_dim"],
                        num_layers=hyperparam["num_layers"],
                        output_size=2, 
                        bidirectional=hyperparam["bidirectional"],
                        dropout=hyperparam["dropout"]
                    ),
                    unmasking_model=bert_model,
                    classifier=RNN(
                        rnn_type=hyperparam["model_type"],
                        embeddings_layer=deepcopy(bert_model.embeddings.word_embeddings),
                        hidden_dim=hyperparam["hidden_dim"],
                        num_layers=hyperparam["num_layers"],
                        output_size=OUTPUT_DIM[data_type],
                        bidirectional=hyperparam["bidirectional"],
                        dropout=hyperparam["dropout"]
                    ),
                    max_seq_length=hyperparam["max_seq_length"]
                )
            logger.info(f"Model created...")
            optimizer = torch.optim.Adam(
                params=[{"params": model.masking_model.parameters()},
                        {"params": model.classifier.parameters()}],
                lr=hyperparam["lr"]       
            )
            loss = WeightedMaskClassificationLoss()
            # Train the model
            trainer = HighwayAugmenterTrainer(
                model=model,
                optimizer=optimizer,
                criterion=loss,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                logger=logger,
                num_epochs=hyperparam["num_epochs"],
            )
            trainer.train()
            if trainer.best_f1_score > best_f1_score:
                best_f1_score = trainer.best_f1_score
                best_accuracy = trainer.best_valid_acc
                best_state_dict = {
                    "model": trainer.best_model,
                    "optimizer": trainer.best_optimizer,
                    "loss": trainer.best_loss,
                }
                best_hyperparam = hyperparam
        # Log val and test results
        logger.info("=" * 60)
        logger.info(f"Finished training model. Best hyperparameters: {best_hyperparam}")
        logger.info("Here is the validation results:")
        HighwayAugmenterTrainer.report_metrics(
            model=best_state_dict["model"], dataloader=val_dataloader, logger=logger
        )
        logger.info(
            "Here is the test results (DO NOT USE THESE RESULTS FOR CHOOSING THE BEST AUGMENTATION):"
        )
        HighwayAugmenterTrainer.report_metrics(
            model=best_state_dict["model"], dataloader=test_dataloader, logger=logger
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
        HighwayAugmenterTrainer.save_checkpoint(
            save_path=model_path,
            model=best_state_dict["model"],
            optimizer=best_state_dict["optimizer"],
            loss=best_state_dict["loss"],
        )
        # Log this to the json
        json_dict = {
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
    # Train models for each augmentation dict
    logger.info("-" * 120)
    logger.info(f"Training models for {data_type}")
    train_models(data_type)


if __name__ == "__main__":
    main()


# Usage: python train_highway_augmenter.py -t articles