MODEL=checkpoints/xsum-copy/checkpoint_best.pt
DATADIR=datasets/xsum/
USERDIR=deps/MASS/MASS-summarization/mass

fairseq-generate $DATADIR --path $MODEL \
    --user-dir $USERDIR --task augmented_summarization_mass \
    --batch-size 64 --beam 4 --min-len 5 --no-repeat-ngram-size 3 --max-len-b 50 --lenpen 1.9 \
    --skip-invalid-size-inputs-valid-test \
    --copy-attn > output.txt

