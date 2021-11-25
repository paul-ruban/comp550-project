import argparse
import os
from pathlib import Path
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
import re


def parse_args():
    # Set up the command line parser
    parser = argparse.ArgumentParser(description="Clean books preprocessing script")
    parser.add_argument("-i", "--input_path", required=True, help="Path of book folder to clean.")
    parser.add_argument(
        "-o", "--output_path", required=True, help="Output path to store cleaned books."
    )
    # Parse arguments
    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    # Do some checks on path
    assert input_path.exists() and input_path.is_dir(), "Input path does not exist or is not a directory."
    # if not output_path.exists():
    #     os.mkdir(output_path)
    assert output_path.is_dir(), "Output path is not a directory."
    return input_path, output_path


def get_all_book_file_paths(input_path):
    book_file_paths = input_path.glob("**/*.epub.txt")
    return book_file_paths


def get_all_books_as_string(book_file_paths):
    books = {}
    print("Reading books :)")
    for book_file_path in tqdm(book_file_paths):
        with open(book_file_path, "r") as book_file:
            books[book_file_path] = book_file.read()
    return books


def preprocess_books(books):
    preprocessed_books = {}
    print("Preprocessing books :)")
    for book_path, book in tqdm(books.items()):
        # Tokenize on sentence level
        sentences = sent_tokenize(book)
        # Tokenize each sentence on word level
        tokenized_sentences = [word_tokenize(sent) for sent in sentences]
        # Join all tokenize words by spaces
        preprocessed_book = [" ".join(sent) for sent in tokenized_sentences]
        preprocessed_books[book_path] = preprocessed_book
    return preprocessed_books


def save_preprocessed_books(preprocessed_book, output_path):
    print("Saving preprocessed books :)")
    for book_path, book in tqdm(preprocessed_book.items()):
        saved_book_path = Path(re.sub(r"(\.txt|\.epub\.txt)", "_cleaned.txt", str(book_path.name)))
        with open(output_path / saved_book_path, "w") as book_file:
            book_file.write("\n".join(book))


def main():
    # Parse args
    input_path, output_path = parse_args()
    # Get book paths
    book_file_paths = get_all_book_file_paths(input_path)
    # Get all books as string
    books = get_all_books_as_string(book_file_paths)
    # Preprocess books
    preprocessed_books = preprocess_books(books)
    # Save preprocessed books
    save_preprocessed_books(preprocessed_books, output_path)


if __name__ == "__main__":
    main()
