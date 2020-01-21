PARADIR=datasets/duc2004/preprocessed-core
PROCDIR=datasets/duc2004/core

mkdir -p $PROCDIR

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --testpref ${PARADIR}/test.1,${PARADIR}/test.2,${PARADIR}/test.3,${PARADIR}/test.4 \
    --destdir $PROCDIR --srcdict datasets/mass-base-uncased/dict.txt --tgtdict datasets/mass-base-uncased/dict.txt \
    --workers 20 \
    --fp16 \


PARADIR=datasets/duc2004/preprocessed-entities
PROCDIR=datasets/duc2004/entities

mkdir -p $PROCDIR

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --testpref ${PARADIR}/test.1,${PARADIR}/test.2,${PARADIR}/test.3,${PARADIR}/test.4 \
    --destdir $PROCDIR --srcdict ${PARADIR}/dict.txt --tgtdict ${PARADIR}/dict.txt \
    --workers 20 \
    --fp16 \