#!/bin/bash
# Run this script with `source installation.sh` in the root directory
# Installs this project's path to your PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Installs all dependencies
pip install -r requirements.txt