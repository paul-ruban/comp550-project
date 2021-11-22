import os
import torch
import random
from datasets import load_dataset


class DataFiles:
    def __init__(self, paths) -> None:
        self.paths = paths
    
    def __len__(self):
        return len(self.paths)
    
    def __iter__(self):
        for path in self.paths:
            yield path
    
    def __str__(self):
        return str(self.paths)

    @classmethod
    def from_dir(cls, dir):
        paths = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

        return cls(paths=paths)
    
    @classmethod
    def from_url_file(cls, url_file):
        with open(url_file) as f:
            urls = [line.rstrip() for line in f.readlines()]

        return cls(paths=urls)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_files, split="all"):
        self.data_files = data_files
        self.dataset = load_dataset(
            path="./dataset_script.py", 
            data_files=data_files, 
            split=split
        )

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

    def map(self, *args, **kwargs):
        self.dataset = self.dataset.map(*args, **kwargs)
        return self
    
    def filter(self, *args, **kwargs):
        return self.dataset.filter(*args, **kwargs)


def remove_empty_lines(examples):
    # non-empty lines
    keep_ids = [i for i, line in enumerate(examples["text"]) if len(line.strip())]

    # remove empty lines
    for feature in examples:
        examples[feature] = [examples[feature][i] for i in keep_ids]
    
    return examples


def tokenize_fn(examples, tokenizer, max_seq_length, fill_to_max=True, sep_token="</s>"):
    # skip tokenization if it is already done
    if examples["text"] and isinstance(examples["text"][0], str):
        examples["text"] = [tokenizer.tokenize(line) for line in examples["text"]]
    
    # split sequences longer than max_seq_length
    i = 0
    while i < len(examples["text"]):
        current_example_length = len(examples["text"][i])
        # -2 for special tokens at the beginning and the end of the sequence
        if current_example_length > max_seq_length - 2:
            example = { feat: examples[feat].pop(i) for feat in examples}
            text = example["text"]
            chunks = [text[i:i+max_seq_length - 2] for i in range(0, len(text), max_seq_length - 2)]
            for j, chunk in enumerate(chunks):
                examples["text"].insert(i, chunk)
                examples["file_idx"].insert(i, example["file_idx"])
                examples["doc_idx"].insert(i, example["doc_idx"])
                examples["line_idx"].insert(i, j)
                i += 1
        else:
            if fill_to_max and (i + 1 < len(examples["text"])):
                next_example_length = len(examples["text"][i+1])

                # if concatenate lines, insert a separator token
                # also make sure lines come from the same file
                if (current_example_length + next_example_length + 1 < max_seq_length and
                    examples["file_idx"][i] == examples["file_idx"][i+1]):
                    next_example = { feat: examples[feat].pop(i) for feat in examples}
                    
                    # join lines, but keep file, document, line ids the same
                    examples["text"][i] += [sep_token] + next_example["text"]
            i += 1
    
    return examples