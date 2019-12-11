PARADIR=datasets/cnndm-constrained-510/k16/preprocessed-core
PROCDIR=datasets/cnndm-constrained-510/k16/core

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref ${PARADIR}/train --validpref ${PARADIR}/valid --testpref ${PARADIR}/test \
    --destdir $PROCDIR --srcdict datasets/mass-base-uncased/dict.txt --tgtdict datasets/mass-base-uncased/dict.txt \
    --workers 20 \
    --fp16 \


PARADIR=datasets/cnndm-constrained-510/k16/preprocessed-entities
PROCDIR=datasets/cnndm-constrained-510/k16/entities

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref ${PARADIR}/train --validpref ${PARADIR}/valid --testpref ${PARADIR}/test \
    --destdir $PROCDIR --srcdict datasets/cnndm-constrained-510/k16/preprocessed-entities/dict.txt --tgtdict datasets/cnndm-constrained-510/k16/preprocessed-entities/dict.txt \
    --workers 20 \
    --fp16 \