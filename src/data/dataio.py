import os
import re
import torch
from datasets import load_dataset

DATASET_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dataset_script.py")


class DataFiles:
    """An abstraction representing a list of file paths that will 
    form a dataset"""
    def __init__(self, paths) -> None:
        self.paths = paths
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __iter__(self) -> str:
        for path in self.paths:
            yield path
    
    def __getitem__(self, i) -> str:
        return self.paths[i]
    
    def __str__(self) -> str:
        return str(self.paths)

    @classmethod
    def from_dir(cls, dir):
        """All .txt files in dir are form FilePaths"""
        paths = [os.path.join(dir, f) for f in os.listdir(dir) 
                if os.path.isfile(os.path.join(dir, f)) and 
                f.endswith(".txt")]
        return cls(paths=paths)
    
    @classmethod
    def from_url_file(cls, url_file):
        """Use a file with urls each on a separate line"""
        with open(url_file) as f:
            urls = [line.rstrip() for line in f.readlines()]

        return cls(paths=urls)


class Dataset(torch.utils.data.Dataset):
    """Represents an iterable dataset that wraps transformers dataset object"""
    def __init__(self, data_files, split="all"):
        self.data_files = data_files
        self.dataset = load_dataset(
            path=DATASET_SCRIPT_PATH, 
            data_files=data_files, 
            split=split
        )

    def __getitem__(self, idx : int) -> dict:
        return self.dataset[idx]

    def __len__(self) -> int:
        return len(self.dataset)

    def map(self, *args, **kwargs):
        self.dataset = self.dataset.map(*args, batched=True, **kwargs)
        return self
    
    def filter(self, *args, **kwargs):
        return self.dataset.filter(*args, **kwargs)


def remove_empty_fn(dataset : Dataset) -> Dataset:
    """Removes empty lines from the dataset."""
    # non-empty lines
    keep_ids = [i for i, line in enumerate(dataset["text"]) if len(line.strip())]
    # remove empty lines
    for feature in dataset:
        dataset[feature] = [dataset[feature][i] for i in keep_ids]

    return dataset


def truncate_fn(dataset: Dataset, tokenizer, max_seq_length : int = 128, fill_to_max : bool = False) -> Dataset:
    """Performs splitting of long lines into several lines by keeping track of their original location.
    Also may merge several lines into one to have fuller data samples.
    
    Parameters
    ----------
    dataset         : Dataset object
    tokenizer       : HuggingFace tokenizer
    max_seq_length  : maximum sequence length accepted by the model we are preparing the dataset for
    fill_to_max     : whether or not merge adjacent short sequnces in one to reduce computation waste
    """
    # add tokens feature
    dataset["tokens"] = [tokenizer.tokenize(line) for line in dataset["text"]]
    
    # split sequences longer than max_seq_length
    i = 0
    while i < len(dataset["tokens"]):
        current_tokens_len = len(dataset["tokens"][i])
        # -2 for special tokens at the beginning and the end of the sequence
        if current_tokens_len > max_seq_length - 2:
            example = {feat : dataset[feat].pop(i) for feat in dataset}
            tokens = example["tokens"]
            chunks = [tokens[i:i+max_seq_length-2] for i in range(0, len(tokens), max_seq_length - 2)]
            for j, chunk in enumerate(chunks):
                dataset["file_id"].insert(i, example["file_id"])
                dataset["line_id"].insert(i, example["line_id"])
                dataset["subline_id"].insert(i, j)
                dataset["tokens"].insert(i, chunk)
                dataset["text"].insert(i, tokenizer.convert_tokens_to_string(dataset["tokens"][i]))
                i += 1
            i -= 1
        else:
            if fill_to_max and (i + 1 < len(dataset["tokens"])):
                next_tokens_len = len(dataset["tokens"][i+1])

                # if concatenate lines, insert a separator token
                # make sure lines come from the same file
                if (current_tokens_len + next_tokens_len + 1 < max_seq_length - 2 and
                    dataset["file_id"][i] == dataset["file_id"][i+1]):
                    next_example = {feat : dataset[feat].pop(i+1) for feat in dataset}
                    
                    # join lines, but keep file, document, line ids the same
                    dataset["text"][i] += tokenizer.sep_token + next_example["text"]
                    dataset["tokens"][i] += [tokenizer.sep_token] + next_example["tokens"]
                    continue
            i += 1
    # remove tokens feature as having them will result in errors when we iterate over batches
    dataset.pop("tokens")
    
    return dataset


def tokenize_fn(dataset : Dataset, tokenizer) -> Dataset:
    """Tokenizes text feature and adds tokens feature to the Dataset."""
    dataset["tokens"] = [tokenizer.tokenize(line) for line in dataset["text"]]

    return dataset


def recover_mask_fn(dataset : Dataset, tokenizer, mask_char='A') -> Dataset:
    """Replace the symbol used to mask words in a file to a mask recognized by the model.
    By convention a single <mask_char> represents a single token. To indicate that multiple tokens 
    were masked we use <number><mask_char>, like 3A -> <mask><mask><mask>.
    Note: this is better to be done before truncating and merging lines. 

    Ex.
    1. "I 'm enjoying A." -> I 'm enjoying <mask>.
    2. "I 'm 2A life." -> I 'm <mask><mask> life.
    """

    dataset["text"] = [
        re.sub(
            pattern=rf'(\d*)({mask_char})', 
            repl=lambda x: tokenizer.mask_token * int(x[1] if x[1] else 1), 
            string=line) 
        for line in dataset["text"]
    ]

    return dataset