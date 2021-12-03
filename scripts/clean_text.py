import argparse
import os
from pathlib import Path
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import re


def parse_args():
    # Set up the command line parser
    parser = argparse.ArgumentParser(description="Clean texts preprocessing script")
    parser.add_argument("-i", "--input_path", required=True, help="Path of text folder to clean.")
    parser.add_argument(
        "-o", "--output_path", required=True, help="Output path to store cleaned texts."
    )
    # Parse arguments
    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = args.output_path
    # Do some checks on path
    assert (
        input_path.exists() and input_path.is_dir()
    ), "Input path does not exist or is not a directory."
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    return input_path, output_path


def get_all_text_file_paths(input_path):
    text_file_paths = input_path.glob("**/*.txt")
    return text_file_paths


def get_all_texts_as_string(text_file_paths):
    texts = {}
    print("Reading texts :)")
    for text_file_path in tqdm(text_file_paths):
        with open(text_file_path, "r", encoding="unicode_escape") as text_file:
            texts[text_file_path] = text_file.read()
    return texts


def preprocess_texts(texts):
    preprocessed_texts = {}
    print("Preprocessing texts :)")
    for text_path, text in tqdm(texts.items()):
        # Tokenize each text
        tokenized_text = word_tokenize(text)
        # Lower case each text
        tokenized_text = [token.lower() for token in tokenized_text]
        # Join all tokenize words by spaces
        preprocessed_text = " ".join(tokenized_text)
        preprocessed_texts[text_path] = preprocessed_text
    return preprocessed_texts


def save_preprocessed_texts(preprocessed_text, output_path):
    print("Saving preprocessed texts :)")
    for text_path, text in tqdm(preprocessed_text.items()):
        saved_text_path = Path(re.sub(r"(\.txt|\.epub\.txt)", "_cleaned.txt", str(text_path.name)))
        with open(output_path / saved_text_path, "w") as text_file:
            text_file.write(text)


def main():
    # Parse args
    input_path, output_path = parse_args()
    # Get text paths
    text_file_paths = get_all_text_file_paths(input_path)
    # Get all texts as string
    texts = get_all_texts_as_string(text_file_paths)
    # Preprocess texts
    preprocessed_texts = preprocess_texts(texts)
    # Save preprocessed texts
    save_preprocessed_texts(preprocessed_texts, output_path)


if __name__ == "__main__":
    main()
