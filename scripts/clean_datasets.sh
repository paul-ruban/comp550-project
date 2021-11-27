#!/bin/bash
data_path='/home/c_spino/comp_550/comp-550-project/data/'
train_folder_='training'
val_folder='validation'
test_folder='testing'
for folder in 'training' 'validation' 'testing'; do
    python clean_text.py -i "${data_path}raw/${folder}/" -o "${data_path}clean/${folder}/"
done