#!/bin/bash
# Script used to create a single dataset from the list of different files
# Pass number of books and train, val, test split with the following default values
num_books=200
train_split=0.8
val_split=0.1
test_split=0.1
# Create a random seed generator (to select the data)
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}
# Absolute path in the drive where all the books are stored (all sizes)
data_path='/content/gdrive/MyDrive/COMP_550_Final_Project/data'
train_path='/dataset/raw/training'
val_path='/dataset/raw/validation'
test_path='/dataset/raw/testing'
data_path_file='../data/raw_dataset_paths.txt'
# Install bc if it doesn't exist, to do floating point operations
command -v bc && echo "bc already installed" || sudo apt install bc
# Use an 80-10-10 split
num_train_books=$(echo "($num_books*$train_split)/1" | bc )
num_val_books=$(echo "($num_books*$val_split)/1" | bc )
num_test_books=$(echo "($num_books*$test_split)/1" | bc )
# Store the paths of each book in the created dataset to a file in the repo
# As a header it contains the splits
find $data_path -name '*.epub.txt' | grep -v Selected | grep -v dataset | shuf --random-source=<(get_seeded_random 42) | head -n $num_books > $data_path_file
# Create the directories if they do not exist, otherwise empty them
for path in $train_path $val_path $test_path; do
    if [ -d "${data_path}${path}" ]; then
        rm -rf $"${data_path}${path}"
    fi
    mkdir "${data_path}${path}"
done
# Fill the directories
echo "Populating $train_path"
cp $(head -n $num_train_books $data_path_file) "${data_path}${train_path}"
echo "Populating $test_path"
cp $(tail -n $num_test_books $data_path_file) "${data_path}${test_path}"
echo "Populating $val_path"
cp $(tail -n $(( $num_val_books + $num_test_books)) $data_path_file | head -n $num_val_books) "${data_path}${val_path}"

