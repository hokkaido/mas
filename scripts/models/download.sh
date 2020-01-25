#!/bin/bash

MODEL_DIR=refactorcheckpoints

mkdir -p ${MODEL_DIR}/mass-base-uncased
echo "Downloading MASS Base Model"

wget "https://modelrelease.blob.core.windows.net/mass/mass-base-uncased.tar.gz" -O ${MODEL_DIR}/mass-base-uncased/mass-base-uncased.tar.gz -q --show-progress

tar -xzf ${MODEL_DIR}/mass-base-uncased/mass-base-uncased.tar.gz -C ${MODEL_DIR}/mass-base-uncased/
