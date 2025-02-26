MODEL=checkpoints/cnndm/checkpoint_best.pt
DATADIR=datasets/cnndm/
USERDIR=deps/MASS/MASS-summarization/mass

fairseq-generate $DATADIR --path $MODEL \
    --user-dir $USERDIR --task augmented_summarization_mass \
    --batch-size 64 --beam 5 --min-len 45 --no-repeat-ngram-size 4 --max-len-b 183 --lenpen 2.0 \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --memory-efficient-fp16 > output.txt
    # --embed-segments-encoder \
    # --embed-segments-decoder \
    # --segment-tokens "." \
    # --max-segments 128 \
