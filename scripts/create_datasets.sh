#!/bin/bash
# Script used to create a single dataset from the list of different files
# Pass number of books and train, val, test split with the following default values
num_books=2225
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
train_path='/training'
val_path='/validation'
test_path='/testing'
data_path='/home/c_spino/comp_550/comp-550-project/data/raw'
data_path_file='/home/c_spino/comp_550/comp-550-project/data/article_paths.txt'
# Install bc if it doesn't exist, to do floating point operations
command -v bc && echo "bc already installed" || sudo apt install bc
# Use an 80-10-10 split
num_train_books=$(echo "($num_books*$train_split)/1" | bc )
num_val_books=$(echo "($num_books*$val_split)/1" | bc )
num_test_books=$(echo "($num_books*$test_split)/1" | bc )
# Store the paths of each book in the created dataset to a file in the repo
# As a header it contains the splits
find $data_path -name '*.txt' | shuf --random-source=<(get_seeded_random 42) | head -n $num_books > $data_path_file
# Create the directories if they do not exist, otherwise empty them
for path in $train_path $val_path $test_path; do
    if [ -d "${data_path}${path}" ]; then
        rm -rf $"${data_path}${path}"
    fi
    mkdir "${data_path}${path}"
done
# Fill the directories
echo "Populating $train_path"
cp $(head -n 1780 $data_path_file) "${data_path}${train_path}"
echo "Populating $test_path"
cp $(tail -n 220 $data_path_file) "${data_path}${test_path}"
echo "Populating $val_path"
cp $(tail -n $(( 225 + 220)) $data_path_file | head -n 225) "${data_path}${val_path}"

