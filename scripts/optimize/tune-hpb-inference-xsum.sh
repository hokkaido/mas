MODEL=checkpoints/xsum/checkpoint_best.pt
DATADIR=datasets/xsum
USERDIR=deps/MASS/MASS-summarization/mass
BATCH_SIZE=64
NUM_WORKERS=2
CONFIG=xsum
RUN_ID=xsum-vanilla


python run_bohb.py $DATADIR --path $MODEL \
    --user-dir $USERDIR --task augmented_summarization_mass \
    --batch-size $BATCH_SIZE \
    --gen-subset valid \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --memory-efficient-fp16 \
    --hpb_config $CONFIG \
    --hpb_run_id $RUN_ID \
    --hpb_min_budget 1 \
    --hpb_max_budget 32 \
    --hpb_n_iterations 32 \
    --hpb_overwrite_run \
    "$@"
