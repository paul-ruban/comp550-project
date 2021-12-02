import copy
import logging
import os
import re
from typing import List, Tuple, Union

import nltk
import numpy as np
import pandas as pd
import sklearn
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

nltk.download("all", quiet=True)
STOPWORDS = set(stopwords.words("english"))
logger = logging.getLogger()


def tokenize_text(
    text: List[str],
    mode: str = "word",
    remove_punctuation: bool = True,
    remove_stopwords: bool = False,
) -> List[List[str]]:
    """Tokenize a list of strings into a list of tokens based on a given mode.
    Also does some post-tokenization operation such as removing punctuation.

    Args:
        text (List[str]): List of strings that represent the texts to analyze.
        mode (str, optional): Tokenizer mode. Defaults to "word".
        remove_punctuation (bool, optional): Whether we should remove punctuation.
        Defaults to True.

    Raises:
        ValueError: If an unrecognized mode is passed.

    Returns:
        List[List[str]]: A a nested list of strings where each string of the inner
        list represents the token of a text.
    """
    assert all(isinstance(elt, str) for elt in text)
    assert isinstance(mode, str)
    assert isinstance(remove_punctuation, bool)
    tokenized_text = []
    if mode == "word":
        tokenized_text = [nltk.word_tokenize(sentence) for sentence in text]
    else:
        raise ValueError("This function does not support any other tokenization mode.")
    if remove_punctuation:
        # Some post-tokenization processing to remove punctuations
        # Keep ! and ? as these might be informative
        punctuation_regex = r"^[.,;\"'():-_`-]+$"  # we keep ! and ?
        tokenized_text = [
            [token for token in tokens if not re.search(punctuation_regex, token)]
            for tokens in tokenized_text
        ]
    if remove_stopwords:
        tokenized_text = [
            [token for token in tokens if token not in STOPWORDS] for tokens in tokenized_text
        ]
    return tokenized_text


def normalize_tokenized_text(tokenized_text: List[List[str]], mode="none") -> List[List[str]]:
    """Normalize inner list of tokens (1 list per "document"/observation) based
    on a certain mode. The possible normalization are none, stemming (via nltk's
    Porter Stemmer) or lemmatization (implemented by nltk).

    Args:
        tokenized_text (List[List[str]]): List of list of strings that represent
        the list of tokens.
        mode (str, optional): Normalization mode. Defaults to "none".

    Raises:
        ValueError: If an unrecognized mode is passed.

    Returns:
        List[List[str]]: A a nested list of tokens where each token has been
        normalized based on a certain normalization scheme.
    """
    assert all(all(isinstance(sub_elt, str) for sub_elt in elt) for elt in tokenized_text)
    text_normalized = []
    if mode == "none":
        return tokenized_text
    elif mode == "stem":
        # All words will be lower cased (the input is lower cased anyway)
        stemmer = nltk.PorterStemmer()
        text_normalized = [[stemmer.stem(token) for token in tokens] for tokens in tokenized_text]
    elif mode == "lemmatize":
        lemmatizer = nltk.WordNetLemmatizer()
        text_normalized = [
            [lemmatizer.lemmatize(token) for token in tokens] for tokens in tokenized_text
        ]
    else:
        raise ValueError(
            "Unrecognized normalization mode, must pick among {none, stem, lemmatize}."
        )
    return text_normalized


def create_N_gram(tokenized_text: List[List[str]], N: int = 1) -> List[List[str]]:
    """N-gram creator which forms inner lists of N-grams in preparation to be fed
    to sklearn's CountVectorizer. Expects an innet list of tokens. (Note we cannot
    simply feed an inner list of tokens to count vectorizer because it will
    try to tokenize for us).

    For example, for N = 2:
    ["This", "is", "an", "example"]
    Should be converted to:
    ["This is", "is an", "an example"]

    Args:
        tokenized_text (List[List[str]]): List of list of strings that represent
        the list of tokens.
        N (str, optional): N in N-gram model. Defaults to 1 i.e. a unigram.

    Raises:
        ValueError: If an unrecognized N is passed.

    Returns:
        List[List[str]]: A a nested list of tokens where each token has been
        normalized based on a certain normalization scheme.
    """
    assert all(all(isinstance(sub_elt, str) for sub_elt in elt) for elt in tokenized_text)
    assert isinstance(N, int) and N >= 1
    N_gram = [(list(ngrams(tokens, N)) if len(tokens) >= N else ()) for tokens in tokenized_text]
    # N_gram becomes an inner list of tuples, this isn't wrong per se but
    # slightly unnatural, we change this to strings
    N_gram = [[" ".join(n_gram) for n_gram in n_grams] for n_grams in N_gram]
    return N_gram


def get_train_test_index(
    text: List[str],
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[list, list, list]:
    """Returns the indices to use for the training, validation and testing dataset.

    Args:
        text (List[str]): List of sentences representing the whole available corpus.
        This corpus will be split into 3 partitions.
        train_size (float, optional): Size of the training set. Defaults to 0.8.
        val_size (float, optional): Size of the validation set. Defaults to 0.1.
        test_size (float, optional): Size of the test set. Defaults to 0.1.
        random_state (int, optional): Random seed. Defaults to 42.

    Returns:
        Tuple[list, list, list]: Lists of indices for the training, validation and test
        set.
    """
    assert all(isinstance(sentence, str) for sentence in text)
    num_list = [train_size, val_size, test_size]
    assert all(isinstance(num, float) for num in num_list)
    assert all(0 < num < 1 for num in num_list)
    assert sum(num_list) == 1
    assert isinstance(random_state, int)
    data_index = np.arange(len(text))
    train_idx, val_idx = train_test_split(
        data_index, train_size=train_size, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        val_idx, train_size=val_size / (1 - train_size), random_state=random_state
    )
    return train_idx, val_idx, test_idx


def get_feature_extractor(
    mode: str = "count",
) -> Union[CountVectorizer, TfidfVectorizer]:
    """Get an sklearn feature extractor which should be able to convert
    a list of N-grams to a feature matrix.

    Args:
        mode (str, optional): What mode the feature returned feature extractor
        should have. Defaults to "count".

    Returns:
        Union[CountVectorizer, TfidfVectorizer]: Sklearn feature extractor which will
        either be a CountVectorizer (for the "count" and "binary" modes) or a
        TfidfVectorizer (for the "frequency" mode).
    """
    assert isinstance(mode, str)
    feature_extractor = None
    # We use lambda x: x as the vectorizers do not support nested lists of strings by default
    # They expect lists of sentences
    if mode == "count":
        feature_extractor = CountVectorizer(analyzer=lambda x: x)
    elif mode == "binary":
        feature_extractor = CountVectorizer(analyzer=lambda x: x, binary=True)
    elif mode == "frequency":
        feature_extractor = TfidfVectorizer(analyzer=lambda x: x)
    return feature_extractor


def create_train_val_set(
    preprocessed_text: List[str],
    labels: list,
    feature_extractor: Union[CountVectorizer, TfidfVectorizer],
    train_idx: list,
    val_idx: list,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """Given a list of preprocessed sentences and their corresponding labels, this
    function uses a feature extractor (either CountVectorizer or TfidfVectorizer) to
    produce a training feature matrix/training label list as well as a validation
    feature matrix and label list.

    Args:
        preprocessed_text (List[List[str]]): List of sentences (represented as lists of
        strings) which have been preprocessed.
        labels (list): List of labels representing sentiment.
        feature_extractor (Union[CountVectorizer, TfidfVectorizer]): Feature extractor
        to apply on training and validation corpus.
        train_idx (list): List of indices indicating the training set.
        val_idx (list): List of indices indicating the validation set.

    Returns:
        Tuple[np.array, np.array, np.array, np.array]: Feature matrices and label arrays for
        both the training and validation set.
    """
    # Validation
    assert isinstance(preprocessed_text, np.ndarray)  # otherwise the indexing won't work
    assert all(all(isinstance(token, str) for token in sentence) for sentence in preprocessed_text)
    assert any(isinstance(feature_extractor, type_) for type_ in [CountVectorizer, TfidfVectorizer])
    # Use the standard sklearn fit/transform workflow
    train_text = preprocessed_text[train_idx]
    X_train = feature_extractor.fit_transform(train_text)
    y_train = labels[train_idx]
    val_text = preprocessed_text[val_idx]
    X_val = feature_extractor.transform(val_text)
    y_val = labels[val_idx]
    # Sanity checks
    assert (
        X_train.shape[0] == y_train.size
    ), f"X_train shape: {X_train.shape}, y_train size: {y_train.size}"
    assert X_val.shape[1] == X_train.shape[1]
    assert X_val.shape[0] == y_val.size
    return X_train, y_train, X_val, y_val


def create_test_set(
    preprocessed_text: List[List[str]],
    labels: list,
    feature_extractor: Union[CountVectorizer, TfidfVectorizer],
    test_idx: list,
) -> Tuple[np.array, np.array]:
    """Given a list of preprocessed sentences and labels, this function creates a test
    set using a feature extractor. Note that the feature extractor must have been initialized
    and fitted on the training set.

    Args:
        preprocessed_text (List[List[str]]): List of preprocessed sentences (represented as list
        of strings).
        labels (list): List of labels representing sentiment.
        feature_extractor (Union[CountVectorizer, TfidfVectorizer]): Feature extractor
        that will be used to convert the test set to a numerical representation. Must have already
        been fit on the training set.
        test_idx (list): List of indices representing the test set.

    Returns:
        Tuple[np.array, np.array]: Feature matrix and label array for the test set.
    """
    assert isinstance(preprocessed_text, np.ndarray)  # otherwise the indexing won't work
    assert all(all(isinstance(token, str) for token in sentence) for sentence in preprocessed_text)
    assert any(isinstance(feature_extractor, type_) for type_ in [CountVectorizer, TfidfVectorizer])
    test_text = preprocessed_text[test_idx]
    X_test = feature_extractor.transform(test_text)
    y_test = labels[test_idx]
    return X_test, y_test


def train_and_evaluate(
    X_train: np.array,
    y_train: np.array,
    X_val: np.array,
    y_val: np.array,
    parameter_grid: sklearn.model_selection.ParameterGrid,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[sklearn.linear_model.LogisticRegression, float]:
    """This function trains and evaluates a logistic regression model by training
    the model parameters on a training set and then validating it against a validation
    set using hyperparamters from a hyperparameter grid. The best model (of the grid
    search) is kept and returned along with its validation accuracy.

    Args:
        X_train (np.array): Training feature matrix N x D where N is number of training data
        point and D is the number of features (which we assume have been created from a feature
        extractor)
        y_train (np.array): Training labels.
        X_val (np.array): Validation feature matrix.
        y_val (np.array): Validation labels.
        parameter_grid (sklearn.model_selection.ParameterGrid): Hyperparameter grid to search
        through.
        random_state (int, optional): Random state used for reproducibility. Defaults to 42.
        verbose (bool, optional): Whether to print the validation accuracy of each
        hyperparameter. Defaults to True.

    Returns:
        Tuple[sklearn.linear_model.LogisticRegression, float]: Returns the best logistic classifier
        as well as the validation accuracy.
    """
    clf = sklearn.linear_model.LogisticRegression(random_state=random_state)
    best_classifier = None
    best_accuracy = 0
    best_parameters = {}
    for parameters in parameter_grid:
        clf.set_params(**parameters)
        clf.fit(X_train, y_train)
        y_train_pred = clf.predict(X_train)
        y_val_pred = clf.predict(X_val)
        train_accuracy = sklearn.metrics.accuracy_score(y_true=y_train, y_pred=y_train_pred)
        val_accuracy = sklearn.metrics.accuracy_score(y_true=y_val, y_pred=y_val_pred)
        if verbose:
            logger.info("-" * 60)
            logger.info(f"Training on parameters {parameters}")
            logger.info(
                f"The train accuracy is {train_accuracy:.3f} and the validation accuracy is {val_accuracy:.3f}."
            )
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_parameters = parameters
            best_classifier = copy.deepcopy(clf)
    logger.info(
        f"The best classifier had parameters {best_parameters} with validation accuracy {best_accuracy:.3f}."
    )
    return best_classifier, best_accuracy


def main():
    # Create log handler #
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("output.log"), logging.StreamHandler()],
    )
    logger.info("Starting preprocessing and training.")

    # Full preprocessing pipeline #
    # Import positive and negative texts
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    polarity_path = os.path.join(cur_dir, "..", "..", "data", "rt-polaritydata")
    text_pos = np.loadtxt(
        os.path.join(polarity_path, "rt-polarity.pos"), dtype=str, delimiter="\n", encoding="latin-1"
    )
    text_neg = np.loadtxt(
        os.path.join(polarity_path, "rt-polarity.pos"), dtype=str, delimiter="\n", encoding="latin-1"
    )
    # Check if the shapes make sense
    assert len(text_pos.shape) == 1 and text_pos.size == 5331
    assert len(text_neg.shape) == 1 and text_neg.size == 5331
    # Compose the two lists of texts
    text = np.hstack([text_pos, text_neg])
    # Consider a positive review to be +1 and a negative review to be 0
    sentiment = np.hstack(
        [np.full(shape=text_pos.size, fill_value=1), np.full(shape=text_neg.size, fill_value=0)]
    )
    assert text.size == sentiment.size == (2 * text_pos.size)
    assert len(text.shape) == len(sentiment.shape) == 1
    # Train, val, test split
    train_idx, val_idx, test_idx = get_train_test_index(
        text=text, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42
    )

    # Baseline estimates, we use countvectorizer but this has no effect on the dummy
    # classification.
    dummy_vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    X_train_dummy = dummy_vectorizer.fit_transform(text[train_idx])
    y_train_dummy = sentiment[train_idx]
    X_val_dummy = dummy_vectorizer.transform(text[val_idx])
    y_val_dummy = sentiment[val_idx]
    # Create the dummy classifiers
    # Random classifier
    dummy_random = DummyClassifier(strategy="uniform", random_state=42)
    dummy_random.fit(X_train_dummy, y_train_dummy)
    y_pred_random = dummy_random.predict(X_val_dummy)
    # Compute the validation accuracies for baselines
    random_accuracy = sklearn.metrics.accuracy_score(y_true=y_val_dummy, y_pred=y_pred_random)
    logger.info(
        f"The random classifier has an accuracy of {random_accuracy:.3f}."
    )

    # Create a grid for searching through different preprocessing decisions
    preprocessing_param_grid = [
        {
            "remove_stopwords": [True, False],
            "normalizer_mode": ["none", "lemmatize", "stem"],
            "N": [1, 2, 3],
            "feature_count_mode": ["count", "binary", "frequency"],
        }
    ]
    preprocessing_parameter_grid = sklearn.model_selection.ParameterGrid(preprocessing_param_grid)
    # Create a grid for hyperparamter settings
    C_range = 10.0 ** (-np.arange(5))
    hyperparam_grid = [
        {"penalty": ["l1"], "C": C_range, "solver": ["liblinear"]},
        {"penalty": ["l2"], "C": C_range},
    ]
    hyperparameter_grid = sklearn.model_selection.ParameterGrid(hyperparam_grid)
    # Running variables to be used in post-processing analysis
    # We will also need to use these for the final test set evaluation
    results_accuracy = []
    best_classifier = None
    best_feature_extractor = None
    best_accuracy = 0
    # Training and validation for each preprocessing setting
    for parameter in preprocessing_parameter_grid:
        # Preprocessing
        logger.info(f"Training and validating with the preprocessing parameters {parameter}.")
        # Tokenize the text
        tokenized_text = tokenize_text(
            text=text,
            mode="word",
            remove_punctuation=True,
            remove_stopwords=parameter["remove_stopwords"],
        )
        # Normalize each tokenized sentence via: no normalization, lemmatization
        # or stemming
        text_tokenized_no_punc = normalize_tokenized_text(
            tokenized_text=tokenized_text, mode=parameter["normalizer_mode"]
        )
        # Convert to N-gram model
        N_gram = create_N_gram(tokenized_text=text_tokenized_no_punc, N=parameter["N"])
        preprocessed_text = np.array([np.array(list_) for list_ in N_gram], dtype="object")
        # Convert to a feature matrix
        feature_extractor = get_feature_extractor(mode=parameter["feature_count_mode"])
        X_train, y_train, X_val, y_val = create_train_val_set(
            preprocessed_text=preprocessed_text,
            labels=sentiment,
            feature_extractor=feature_extractor,
            train_idx=train_idx,
            val_idx=val_idx,
        )
        # Train and evaluate
        clf, val_accuracy = train_and_evaluate(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            parameter_grid=hyperparameter_grid,
            verbose=False,
        )
        # Record the best classifier for test set evaluation
        if val_accuracy > best_accuracy:
            best_classifier = clf
            best_feature_extractor = feature_extractor
            best_accuracy = val_accuracy
        # Store result in dictionary for analysis
        result = {**parameter, **{"accuracy": val_accuracy}}
        results_accuracy += [result]

    # Analysis of preprocessing decisions #
    # Get some analysis metrics
    df = pd.DataFrame(results_accuracy)
    # How many of the {"N", "feature_count_mode", "normalizer_mode"} variations have
    # remove_stopwords = True with a higher validation accuracy than remove_stopwords = False
    preprocessing_parameters = set(preprocessing_parameter_grid.param_grid[0].keys())
    for preprocessing_parameter in preprocessing_parameters:
        free_parameters = list(preprocessing_parameters - {preprocessing_parameter})
        idx = df.groupby(free_parameters)["accuracy"].idxmax().values
        logger.info(f"Distribution of highest validation accuracy for {preprocessing_parameter}")
        logger.info(df.iloc[idx].groupby(preprocessing_parameter).count()["accuracy"])
        # We also would like to compute the range of these differences (because maybe
        # a decision is always better but not by very much) to so we also compute the min
        # accuracies and calculate some statistics on the difference between the max and min
        # accuracy
        idx_min = df.groupby(free_parameters)["accuracy"].idxmin().values
        differences = df.loc[idx]["accuracy"].values - df.loc[idx_min]["accuracy"].values
        logger.info("Statistics about the the differences between the max and min accuracy: ")
        logger.info(f"Mean: {differences.mean()}")
        logger.info(f"Min: {differences.min()}")
        logger.info(f"Max: {differences.max()}")
        logger.info(f"Std: {differences.std()}")

    # Final evaluation on test set #
    # Preprocess test set according to optimal preprocessing sequence
    tokenized_text = tokenize_text(
        text=text, mode="word", remove_punctuation=True, remove_stopwords=False
    )
    text_normalized = normalize_tokenized_text(tokenized_text=tokenized_text, mode="stem")
    N_gram = create_N_gram(tokenized_text=text_normalized, N=1)
    preprocessed_text = np.array([np.array(list_) for list_ in N_gram], dtype="object")
    # Use the feature extractor and classifier which have already been fit on the optimal
    # preprocessing decision sequence
    X_test, y_test = create_test_set(preprocessed_text, sentiment, best_feature_extractor, test_idx)
    y_pred = best_classifier.predict(X_test)
    logger.info(
        f"The final test set accuracy is {sklearn.metrics.accuracy_score(y_true=y_test, y_pred=y_pred):.3f}"
    )
    # Print some evaluation metrics
    logger.info(sklearn.metrics.confusion_matrix(y_true=y_test, y_pred=y_pred))
    logger.info(sklearn.metrics.classification_report(y_true=y_test, y_pred=y_pred))
    # See a few of the models mistakes
    misclassified = np.where(y_test != y_pred)
    logger.info(
        np.column_stack([text[test_idx][misclassified], sentiment[test_idx][misclassified]])[:5]
    )

    logger.info("Done!")


if __name__ == "__main__":
    main()
