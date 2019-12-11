
PARADIR=datasets/cnndm-constrained-510/k16/preprocessed-core
PARADIR_GEN=datasets/cnndm-constrained-510/k16-generated/preprocessed-core
PROCDIR_GEN=datasets/cnndm-constrained-510/k16-generated/core

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref ${PARADIR}/train --validpref ${PARADIR}/valid --testpref ${PARADIR_GEN}/test \
    --destdir $PROCDIR_GEN --srcdict datasets/mass-base-uncased/dict.txt --tgtdict datasets/mass-base-uncased/dict.txt \
    --workers 20 \
    --fp16 \

PARADIR=datasets/cnndm-constrained-510/k16/preprocessed-entities
PARADIR_GEN=datasets/cnndm-constrained-510/k16-generated/preprocessed-entities
PROCDIR_GEN=datasets/cnndm-constrained-510/k16-generated/entities

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref ${PARADIR}/train --validpref ${PARADIR}/valid --testpref ${PARADIR_GEN}/test \
    --destdir $PROCDIR_GEN --srcdict datasets/cnndm-constrained-510/k16-generated/preprocessed-entities/dict.src.txt --tgtdict datasets/cnndm-constrained-510/k16-generated/preprocessed-entities/dict.tgt.txt \
    --workers 20 \
    --fp16 \