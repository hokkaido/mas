MODEL=checkpoints/checkpoint_best.pt
DATADIR=datasets/gigaword-augmented/
USERDIR=deps/MASS/MASS-summarization/mass

fairseq-generate $DATADIR --path $MODEL \
    --user-dir $USERDIR --task augmented_summarization_mass \
    --batch-size 64 --beam 6 --min-len 4 --no-repeat-ngram-size 5 \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --embed-entities \
    --memory-efficient-fp16 \
    --lenpen 0.8 > output.txt

