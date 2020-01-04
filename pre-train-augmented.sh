WARMUP_UPDATES=4000
PEAK_LR=0.0005
TOTAL_UPDATES=125000
MAX_TOKENS=4096
UPDATE_FREQ=8

fairseq-train datasets/cnndm-augmented-510/ \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_summarization_mass --arch summarization_mass_base \
    --criterion masked_lm \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --max-tokens $MAX_TOKENS \
    --update-freq $UPDATE_FREQ \
    --ddp-backend=no_c10d \
    --max-source-positions 512 --max-target-positions 512 \
    --fp16 \
    --memory-efficient-fp16 \
    --skip-invalid-size-inputs-valid-test \
    --load-from-pretrained-model datasets/mass-base-uncased/mass-base-uncased.pt \
