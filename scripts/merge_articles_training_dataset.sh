#!/bin/bash
clean_folder_path="/home/c_spino/comp-550-copy/comp-550-project/data/articles/clean_articles/"
augmentation_path="/home/c_spino/comp-550-copy/comp-550-project/data/articles/augmentation/"
for class in business entertainment politics sport tech; do
    new_file="${augmentation_path}${class}.txt"
    for folder in training/ validation/ testing/; do
        for file in "${clean_folder_path}${folder}${class}"*; do
            (cat $file ; echo ) >> $new_file
        done
    done
done