fairseq-train datasets/cnndm/ \
    --user-dir deps/MASS/MASS-summarization/mass --task summarization_mass --arch summarization_mass_base \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 0.0005 --min-lr 1e-09 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --weight-decay 0.0 \
    --criterion debug_criterion \
    --update-freq 8 --max-tokens 4096 \
    --ddp-backend=no_c10d --max-epoch 25 \
    --max-source-positions 512 --max-target-positions 512 \
    --fp16 \
    --memory-efficient-fp16 \
    --skip-invalid-size-inputs-valid-test \
    --copy-attn \
    --load-from-pretrained-model datasets/mass-base-uncased/mass-base-uncased.pt \