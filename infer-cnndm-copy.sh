MODEL=checkpoints/cnndm-copy/checkpoint_best.pt
DATADIR=datasets/cnndm-augmented-510/
USERDIR=deps/MASS/MASS-summarization/mass

CUDA_LAUNCH_BLOCKING=1 fairseq-generate $DATADIR --path $MODEL \
    --user-dir $USERDIR --task augmented_summarization_mass \
    --batch-size 64 --beam 5 --min-len 45 --no-repeat-ngram-size 4 --max-len-b 183 --lenpen 2.0 \
    --skip-invalid-size-inputs-valid-test \
    --copy-attn > output.txt

