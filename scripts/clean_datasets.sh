#!/bin/bash
data_path='/content/gdrive/MyDrive/COMP_550_Final_Project/data/dataset/'
train_folder_='training'
val_folder='validation'
test_folder='testing'
for folder in $train_folder $val_folder $test_folder; do
    python clean_books.py -i "${data_path}raw/${folder}/" -o "${data_path}clean/${folder}/"
done