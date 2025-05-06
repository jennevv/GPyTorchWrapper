#!/bin/bash

LOCAL_DIR=$(pwd)

# Training
docker run \
    -v "$LOCAL_DIR":/app \
    image \
    conda run -n env "$@" \
    python training_gpytorch.py \
    -i /app/example/example_data.csv \
    -f csv \
    -o /app/model \
    -d /app/example \
    -c /app/example/config_example.yaml

# Plotting
docker run \
    -v "$LOCAL_DIR":/app \
    image \
    conda run -n env "$@" \
    python /app/example/plot.py

