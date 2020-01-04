MODEL=checkpoints/cnndm-entities-encoder-copy/checkpoint_best.pt
DATADIR=datasets/cnndm-augmented-510
USERDIR=deps/MASS/MASS-summarization/mass
BATCH_SIZE=32
NUM_WORKERS=2

python gen-hpb-search.py $DATADIR --path $MODEL \
    --user-dir $USERDIR --task augmented_summarization_mass \
    --batch-size $BATCH_SIZE \
    --gen-subset valid \
    --skip-invalid-size-inputs-valid-test \
    --embed-entities-encoder \
    --fp16 \
    --memory-efficient-fp16 \
    --hpb_worker

