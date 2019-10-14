PARADIR=datasets/cnndmsent/para-400
PROCDIR=datasets/cnndmsent/processed-400

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref ${PARADIR}/train --validpref ${PARADIR}/valid --testpref ${PARADIR}/test \
    --destdir $PROCDIR --srcdict datasets/mass-base-uncased/dict.txt --tgtdict datasets/mass-base-uncased/dict.txt \
    --workers 20 \
    --fp16 \