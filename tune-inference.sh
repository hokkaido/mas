MODEL=checkpoints/checkpoint_best.pt
DATADIR=datasets/gigaword-augmented
USERDIR=deps/MASS/MASS-summarization/mass

python gen-search.py $DATADIR --path $MODEL \
    --user-dir $USERDIR --task augmented_summarization_mass \
    --batch-size 200 \
    --skip-invalid-size-inputs-valid-test \
    --embed-entities-encoder \
    --fp16 \
    --memory-efficient-fp16 \

