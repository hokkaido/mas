MODEL=checkpoints/checkpoint_best.pt
DATADIR=datasets/cnndm/
USERDIR=deps/MASS/MASS-summarization/mass

python gen-search.py $DATADIR --path $MODEL \
    --user-dir $USERDIR --task segmented_summarization_mass \
    --batch-size 200 \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --memory-efficient-fp16 \

