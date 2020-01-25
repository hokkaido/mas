#!/bin/bash

PARADIR=datasets/xsum/preprocessed-core
PROCDIR=datasets/xsum/core

mkdir -p $PROCDIR

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --testpref ${PARADIR}/test \
    --destdir $PROCDIR --srcdict datasets/shared/mass-base-uncased.dict.txt --tgtdict datasets/shared/mass-base-uncased.dict.txt \
    --workers 20 \
    --fp16 \


PARADIR=datasets/xsum/preprocessed-entities
PROCDIR=datasets/xsum/entities

mkdir -p $PROCDIR

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref ${PARADIR}/train --validpref ${PARADIR}/validation --testpref ${PARADIR}/test \
    --destdir $PROCDIR --srcdict datasets/shared/entities.dict.txt --tgtdict datasets/shared/entities.dict.txt \
    --workers 20 \
    --fp16 \