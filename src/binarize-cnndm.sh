PARADIR=datasets/cnndm-augmented-511/preprocessed-core
PROCDIR=datasets/cnndm-augmented-511/core

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref ${PARADIR}/train --validpref ${PARADIR}/valid --testpref ${PARADIR}/test \
    --destdir $PROCDIR --srcdict datasets/mass-base-uncased/dict.txt --tgtdict datasets/mass-base-uncased/dict.txt \
    --workers 20 \
    --fp16 \


PARADIR=datasets/cnndm-augmented-511/preprocessed-entities
PROCDIR=datasets/cnndm-augmented-511/entities

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref ${PARADIR}/train --validpref ${PARADIR}/valid --testpref ${PARADIR}/test \
    --destdir $PROCDIR --srcdict datasets/cnndm-augmented-511/preprocessed-entities/dict.txt --tgtdict datasets/cnndm-augmented-511/preprocessed-entities/dict.txt \
    --workers 20 \
    --fp16 \