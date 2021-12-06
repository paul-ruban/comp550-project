import os
import re

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


class WordTokenizer:
    def __init__(self, remove_punctuation=True):
        self.remove_punctuation = remove_punctuation

    def __call__(self, doc):
        if not self.remove_punctuation:
            return [t for t in word_tokenize(doc)]
        else:
            return [
                t for t in word_tokenize(doc) if not re.search(r"[\.,\?\!\:\;\(\)\[\]\{\}]*", t)
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
                if not re.search(r"[\.,\?\!\:\;\(\)\[\]\{\}]*", t)
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
                if not re.search(r"[\.,\?\!\:\;\(\)\[\]\{\}]*", t)
            ]


cur_dir = os.path.dirname(os.path.abspath(__file__))

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
    },
}

PREPROCESSING_GRID = {
    "tokenize_type": ["word", "lemma", "stem"],
    "remove_punctuation": [True, False],
    "remove_stopwords": [True, False],
    "ngram_range": [(1, 1), (1, 2), (1, 3)],
    "count_type": ["raw_counts", "binary_counts", "tfidf"],
}


def create_preprocessing_pipeline_list(
    tokenize_type: str = "word",
    remove_punctuation: bool = True,
    remove_stopwords: bool = True,
    ngram_range: tuple = (1, 1),
    count_type="raw_counts",
) -> list:
    assert tokenize_type in {"word", "lemma", "stem"}
    assert isinstance(remove_punctuation, bool)
    assert isinstance(remove_stopwords, bool)
    assert (
        isinstance(ngram_range, tuple)
        and len(ngram_range) == 2
        and all(elt in {1, 2, 3} for elt in ngram_range)
    )
    assert count_type in {"raw_counts", "binary_counts", "tfidf"}

    preprocessing_pipeline = []
    if tokenize_type == "word":
        tokenizer = WordTokenizer(remove_punctuation=remove_punctuation)
    elif tokenize_type == "lemma":
        tokenizer = LemmaTokenizer(remove_punctuation=remove_punctuation)
    elif tokenize_type == "stem":
        tokenizer = StemmerTokenizer(remove_punctuation=remove_punctuation)
    preprocessing_pipeline.append(("tokenizer", tokenizer))

    if count_type == "raw_counts":
        vectorizer = CountVectorizer(
            tokenize=tokenizer,
            ngram_range=ngram_range,
            stop_words=None if not remove_stopwords else "english",
        )
    elif count_type == "binary_counts":
        vectorizer = CountVectorizer(
            tokenize=tokenizer,
            ngram_range=ngram_range,
            binary=True,
            stop_words=None if not remove_stopwords else "english",
        )
    elif count_type == "tfidf":
        vectorizer = TfidfVectorizer(
            tokenize=tokenizer,
            ngram_range=ngram_range,
            stop_words=None if not remove_stopwords else "english",
        )
    preprocessing_pipeline.append(("vectorizer", vectorizer))

    return preprocessing_pipeline


def create_training_pipeline(
    preprocessing_list: list, model_type: str, model_params: dict
) -> Pipeline:
    assert model_type in {"naive_bayes", "logistic", "random_forest", "mlp", "svm"}
    assert isinstance(model_params, dict)

    if model_type == "naive_bayes":
        model = MultinomialNB(**model_params)
    elif model_type == "logistic":
        model = LogisticRegression(**model_params)
    elif model_type == "boosted_tree":
        model = RandomForestClassifier(**model_params)
    elif model_type == "svm":
        model = SVC(**model_params)
    elif model_type == "mlp":
        model = MLPClassifier(**model_params)

    return Pipeline(preprocessing_list + [("model", model)])


def main():
    pass


if __name__ == "__main__":
    main()
