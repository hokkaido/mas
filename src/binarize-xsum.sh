PARADIR=datasets/duc2004/preprocessed-core
PROCDIR=datasets/duc2004/core

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --testpref ${PARADIR}/test \
    --destdir $PROCDIR --srcdict datasets/mass-base-uncased/dict.txt --tgtdict datasets/mass-base-uncased/dict.txt \
    --workers 20 \
    --fp16 \


PARADIR=datasets/xsum/preprocessed-entities
PROCDIR=datasets/xsum/entities

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref ${PARADIR}/train --validpref ${PARADIR}/validation --testpref ${PARADIR}/test \
    --destdir $PROCDIR --srcdict datasets/xsum/preprocessed-entities/dict.txt --tgtdict datasets/xsum/preprocessed-entities/dict.txt \
    --workers 20 \
    --fp16 \