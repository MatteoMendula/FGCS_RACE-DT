#!/bin/bash

# Initialize conda (adjust the path if needed)
source ~/anaconda3/etc/profile.d/conda.sh  # or ~/miniconda3/etc/profile.d/conda.sh

conda init

# Activate the environment
conda activate reservoir

# list dataset folders
assets=(
    "FD001"
    "FD002"
    "FD003"
    "FD004"
)

# Loop through each folder and run the script with the folder as first argument and the layer as second argument
for folder in "${assets[@]}"; do
    echo "Running script for folder: $folder with layer: $layer"
    # Run your script here, passing the folder and layer as arguments
    python torch_LSTM_gpu.py "$folder"
done

# Loop through each folder and run the script with the folder as first argument and the layer as second argument
for folder in "${assets[@]}"; do
    echo "Running script for folder: $folder with layer: $layer"
    # Run your script here, passing the folder and layer as arguments
    python torch_LSTM_cpu.py "$folder"
done
