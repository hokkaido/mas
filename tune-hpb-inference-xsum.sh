MODEL=checkpoints/xsum-entities-encoder/checkpoint_best.pt
DATADIR=datasets/xsum
USERDIR=deps/MASS/MASS-summarization/mass
BATCH_SIZE=64
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

