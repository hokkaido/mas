MODEL=checkpoints/checkpoint_best.pt
DATADIR=datasets/cnndm-augmented-510/
USERDIR=deps/MASS/MASS-summarization/mass

fairseq-generate $DATADIR --path $MODEL \
    --user-dir $USERDIR --task augmented_summarization_mass \
    --batch-size 64 --beam 5 --min-len 50 --no-repeat-ngram-size 3 \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --memory-efficient-fp16 \
    --lenpen 1.0 > output.txt

