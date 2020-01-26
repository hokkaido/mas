MODEL=checkpoints/xsum-ft-on-joint-pretrained/checkpoint_best.pt
DATADIR=datasets/xsum/
USERDIR=deps/MASS/MASS-summarization/mass

fairseq-generate $DATADIR --path $MODEL \
    --user-dir $USERDIR --task augmented_summarization_mass \
    --batch-size 64 --beam 4 --min-len 5 --no-repeat-ngram-size 3 --max-len-b 50 --lenpen 1.9 \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --memory-efficient-fp16 > output.txt

