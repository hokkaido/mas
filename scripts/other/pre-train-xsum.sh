WARMUP_UPDATES=4000
PEAK_LR=0.0001
TOTAL_UPDATES=300000
MAX_TOKENS=4096
UPDATE_FREQ=8

fairseq-train datasets/xsum/ \
    --user-dir deps/MASS/MASS-summarization/mass --task masked_summarization_mass --arch summarization_mass_base \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --lr 0.0001  --max-update $TOTAL_UPDATES \
    --criterion label_smoothed_cross_entropy \
    --max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ \
    --ddp-backend=no_c10d \
    --max-source-positions 512 --max-target-positions 512 \
    --fp16 \
    --memory-efficient-fp16 \
    --skip-invalid-size-inputs-valid-test \
    --load-from-pretrained-model checkpoints/mass-base-uncased/mass-base-uncased.pt \
