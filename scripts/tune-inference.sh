MODEL=checkpoints/cnndm-entities-encoder/checkpoint_best.pt
DATADIR=datasets/cnndm-augmented-510
USERDIR=deps/MASS/MASS-summarization/mass

python gen-search.py $DATADIR --path $MODEL \
    --user-dir $USERDIR --task augmented_summarization_mass \
    --batch-size 32 \
    --gen-subset valid \
    --skip-invalid-size-inputs-valid-test \
    --embed-entities-encoder \
    --fp16 \
    --memory-efficient-fp16 \

