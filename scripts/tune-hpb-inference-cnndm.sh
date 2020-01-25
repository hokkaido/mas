MODEL=checkpoints/cnndm-entities-encoder-copy/checkpoint_best.pt
DATADIR=datasets/cnndm-augmented-510
USERDIR=deps/MASS/MASS-summarization/mass
BATCH_SIZE=32
NUM_WORKERS=2
CONFIG=cnndm
RUN_ID=cnndm-entities-encoder-copy

python gen-hpb-search.py $DATADIR --path $MODEL \
    --user-dir $USERDIR --task augmented_summarization_mass \
    --batch-size $BATCH_SIZE \
    --gen-subset valid \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --embed-entities-encoder \
    --copy-attn \
    --memory-efficient-fp16 \
    --hpb_config $CONFIG \
    --hpb_run_id $RUN_ID \
    --hpb_min_budget 1 \
    --hpb_max_budget 32 \
    --hpb_n_iterations 16 \
    --hpb_overwrite_run \
    "$@"


