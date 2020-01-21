MODEL=checkpoints/cnndm-reference/checkpoint_best.pt
DATADIR=datasets/cnndm-augmented-510/
USERDIR=deps/MASS/MASS-summarization/mass

python vis-model-probs.py $DATADIR --path $MODEL \
    --user-dir $USERDIR --task augmented_summarization_mass \
    --skip-invalid-size-inputs-valid-test \
    --cpu \


    # --embed-segments-encoder \
    # --embed-segments-decoder \
    # --segment-tokens "." \
    # --max-segments 128 \
