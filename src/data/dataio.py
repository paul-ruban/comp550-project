import os
import torch
import random
from datasets import load_dataset

DATASET_SCRIPT_PATH = "/home/pavlo/comp-550/comp-550-project/src/data/dataset_script.py"


class DataFiles:
    """An abstraction representing a list of file paths that will 
    form a dataset"""
    def __init__(self, paths) -> None:
        self.paths = paths
    
    def __len__(self):
        return len(self.paths)
    
    def __iter__(self):
        for path in self.paths:
            yield path
    
    def __getitem__(self, i):
        return self.paths[i]
    
    def __str__(self):
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

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def map(self, *args, **kwargs):
        self.dataset = self.dataset.map(*args, batched=True, **kwargs)
        return self
    
    def filter(self, *args, **kwargs):
        return self.dataset.filter(*args, **kwargs)


def remove_empty_fn(examples):
    # non-empty lines
    keep_ids = [i for i, line in enumerate(examples["text"]) if len(line.strip())]
    # remove empty lines
    for feature in examples:
        examples[feature] = [examples[feature][i] for i in keep_ids]

    return examples


def tokenize_fn(examples, tokenizer, max_seq_length, fill_to_max=False):
    examples["tokens"] = [tokenizer.tokenize(line) for line in examples["text"]]
    
    # split sequences longer than max_seq_length
    i = 0
    while i < len(examples["tokens"]):
        current_tokens_len = len(examples["tokens"][i])
        # -2 for special tokens at the beginning and the end of the sequence
        if current_tokens_len > max_seq_length - 2:
            example = {feat : examples[feat].pop(i) for feat in examples}
            tokens = example["tokens"]
            chunks = [tokens[i:i+max_seq_length-2] for i in range(0, len(tokens), max_seq_length - 2)]
            for j, chunk in enumerate(chunks):
                examples["file_id"].insert(i, example["file_id"])
                examples["line_id"].insert(i, example["line_id"])
                examples["subline_id"].insert(i, j)
                examples["tokens"].insert(i, chunk)
                examples["text"].insert(i, tokenizer.convert_tokens_to_string(examples["tokens"][i]))
                i += 1
            i -= 1
        else:
            if fill_to_max and (i + 1 < len(examples["tokens"])):
                next_tokens_len = len(examples["tokens"][i+1])

                # if concatenate lines, insert a separator token
                # make sure lines come from the same file
                if (current_tokens_len + next_tokens_len + 1 < max_seq_length - 2 and
                    examples["file_id"][i] == examples["file_id"][i+1]):
                    next_example = {feat : examples[feat].pop(i+1) for feat in examples}
                    
                    # join lines, but keep file, document, line ids the same
                    examples["text"][i] += tokenizer.sep_token + next_example["text"]
                    examples["tokens"][i] += [tokenizer.sep_token] + next_example["tokens"]
                    continue
            i += 1
    
    return examples