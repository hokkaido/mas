#!/bin/bash
DATA_DIR=refactorsets
INPUT_DIR=${DATA_DIR}/cnndm/preprocessed
OUTPUT_DIR=${DATA_DIR}/cnndm-constrained/preprocessed
CLASSIFIER_OUTPUT_DIR=${DATA_DIR}/cnndm-constrained/labels

mkdir -p ${OUTPUT_DIR}
mkdir -p ${CLASSIFIER_OUTPUT_DIR}

python constrain.py --k 16 --output-path=${OUTPUT_DIR}/valid.src \
    --article-path=${INPUT_DIR}/valid.src --abstract-path=${INPUT_DIR}/valid.tgt \
    --classifier-output-path=${CLASSIFIER_OUTPUT_DIR}/labels.valid.txt
cp ${INPUT_DIR}/valid.tgt ${OUTPUT_DIR}/valid.tgt 

python constrain.py --k 16 --output-path=${OUTPUT_DIR}/train.src \
        --article-path=${INPUT_DIR}/train.src --abstract-path=${INPUT_DIR}/train.tgt \
        --classifier-output-path=${CLASSIFIER_OUTPUT_DIR}/labels.train.txt
cp ${INPUT_DIR}/train.tgt ${OUTPUT_DIR}/train.tgt 