fairseq-train datasets/xsum/ \
    --user-dir deps/MASS/MASS-summarization/mass --task augmented_summarization_mass --arch summarization_mass_base \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --min-lr 1e-09 \
    --lr-scheduler polynomial_decay \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --update-freq 8 --max-tokens 4096 \
    --ddp-backend=no_c10d --max-epoch 25 \
    --max-source-positions 512 --max-target-positions 512 \
    --fp16 \
    --memory-efficient-fp16 \
    --skip-invalid-size-inputs-valid-test \
    --load-from-pretrained-model checkpoints/cnndm-pretrained--/checkpoint_best.pt \


# fairseq-train datasets/cnndm/ \
#     --user-dir deps/MASS/MASS-summarization/mass --task augmented_summarization_mass --arch summarization_mass_base \
#     --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#     --lr 0.00035 --min-lr 1e-09 \
#     --lr-scheduler polynomial_decay \
#     --weight-decay 0.0 \
#     --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#     --update-freq 8 --max-tokens 4096 \
#     --ddp-backend=no_c10d --max-epoch 5 \
#     --max-source-positions 512 --max-target-positions 512 \
#     --fp16 \
#     --memory-efficient-fp16 \
#     --skip-invalid-size-inputs-valid-test \
#     --load-from-pretrained-model checkpoints/cnndm-pretrained-batch-2/checkpoint2.pt \
