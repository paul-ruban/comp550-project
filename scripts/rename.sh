#/bin/bash
cd /home/c_spino/comp_550/comp-550-project/data/bbc/tech
for file in *.txt; do
    mv "$file" "tech_${file}"
done