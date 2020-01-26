#!/bin/bash

DATA_DIR=refactorsets
INPUT_DIR=${DATA_DIR}/cnndm/preprocessed
OUTPUT_DIR=${DATA_DIR}/cnndm-constrained/preprocessed
MODEL_DIR=outputs/checkpoint-1500

mkdir -p ${OUTPUT_DIR}

python constrain_with_selector.py --k 16 --model xlnet --model-dir=${MODEL_DIR} \
                                  --article-path ${INPUT_DIR}/test.src \
                                  --output-path ${OUTPUT_DIR}/test.src

cp ${INPUT_DIR}/test.tgt ${OUTPUT_DIR}/test.tgt 
