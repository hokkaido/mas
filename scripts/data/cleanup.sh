#!/bin/bash

DATA_DIR=datasets

echo "Cleaning up CNN-DM..."
python cleanup.py --config cnndm --input-dir ${DATA_DIR}/cnndm/raw --output-dir ${DATA_DIR}/cnndm/preprocessed

echo "Cleaning up XSum..."
python cleanup.py --config xsum --input-dir ${DATA_DIR}/xsum/raw --output-dir ${DATA_DIR}/xsum/preprocessed

echo "Cleaning up DUC2004.."
python cleanup.py --config duc2004 --input-dir ${DATA_DIR}/duc2004/raw --output-dir ${DATA_DIR}/duc2004/preprocessed