#!/bin/bash
# Run this script with `source installation.sh` in the root directory
# Installs this project's path to your PYTHONPATH
project_dir=$(pwd)
if [[ ${PYTHONPATH} != *"${project_dir}"* ]]; then
    export PYTHONPATH="${PYTHONPATH}:${project_dir}"
    echo "Project directory added to PYTHONPATH: ${project_dir}"
fi
# Make sure you call the project the same thing in your drive
echo alias dp="'echo /content/gdrive/MyDrive/COMP_550_Final_Project/data/'" >> ~/.bashrc
source ~/.bashrc
# Installs all dependencies
pip install -r requirements.txt