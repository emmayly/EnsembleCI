#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define your commands in an array
commands=(
    "python3 carbonForecast.py EPE d GBM l",
)

# Loop through each command and execute it
for cmd in "${commands[@]}"; do
    echo "Executing: $cmd"
    eval "$cmd"
    echo "Finished: $cmd"
    echo "----------------------------------------"
done

echo "All commands executed successfully!"
