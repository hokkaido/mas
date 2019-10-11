fairseq-preprocess \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_s2s \
    --source-lang src --target-lang tgt \
    --trainpref datasets/cnndmsent/para/train --validpref datasets/cnndmsent/para/valid --testpref datasets/cnndmsent/para/test \
    --destdir datasets/cnndmsent/processed --srcdict datasets/mass-base-uncased/dict.txt --tgtdict datasets/mass-base-uncased/dict.txt \
    --workers 20