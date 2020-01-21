# PARADIR=datasets/gigaword-augmented/preprocessed-core
# PROCDIR=datasets/gigaword-augmented/core

# fairseq-preprocess \
#     --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
#     --source-lang src --target-lang tgt \
#     --trainpref ${PARADIR}/train --validpref ${PARADIR}/valid --testpref ${PARADIR}/test \
#     --destdir $PROCDIR --srcdict datasets/mass-base-uncased/dict.txt --tgtdict datasets/mass-base-uncased/dict.txt \
#     --workers 20 \
#     --fp16 \



PARADIR=datasets/gigaword-augmented/preprocessed-entities
PROCDIR=datasets/gigaword-augmented/entities

fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref ${PARADIR}/train --validpref ${PARADIR}/valid --testpref ${PARADIR}/test \
    --destdir $PROCDIR --srcdict datasets/gigaword-augmented/preprocessed-entities/dict.txt --tgtdict datasets/gigaword-augmented/preprocessed-entities/dict.txt \
    --workers 20 \
    --fp16 \