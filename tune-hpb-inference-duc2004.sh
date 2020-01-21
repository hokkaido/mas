MODEL=checkpoints/cnndm-reference/checkpoint_best.pt
DATADIR=datasets/duc2004
USERDIR=deps/MASS/MASS-summarization/mass
BATCH_SIZE=25
NUM_WORKERS=2
CONFIG=duc2004
RUN_ID=duc2004-cnndm-test

python gen-hpb-search.py $DATADIR --path $MODEL \
    --user-dir $USERDIR --task augmented_summarization_mass \
    --batch-size $BATCH_SIZE \
    --gen-subset test2 \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --memory-efficient-fp16 \
    --hpb_config $CONFIG \
    --hpb_run_id $RUN_ID \
    --hpb_min_budget 1 \
    --hpb_max_budget 20 \
    --hpb_n_iterations 50 \
    --hpb_overwrite_run \
    --hpb_metric 'ROUGE-1-R (avg)'
    "$@"


