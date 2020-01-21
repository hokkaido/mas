#!/bin/bash

EXPERIMENT=exp4
TASK=augmented_summarization_mass
EVALDIR=eval
MODELDIR=checkpoints
DATADIR=datasets/duc2004
USERDIR=deps/MASS/MASS-summarization/mass
BATCH_SIZE=64
BEAM_PARAMS="--beam 4 --min-len 18 --max-len-a 0 --no-repeat-ngram-size 3 --max-len-b 18"

declare -A MODELS

MODELS=(
    ['xsum-vanilla']=''
    ['xsum-entities-encoder']="--embed-entities-encoder"
    ['xsum-entities-encoder-segments-encoder']="--embed-entities-encoder --embed-segments-encoder --segment-tokens . --max-segments 128"
    ['xsum-segments-encoder']="--embed-segments-encoder --segment-tokens . --max-segments 128"
)

# Save fairseq params for experiment
echo "TASKR: $TASK" | tee -a $EVALDIR/$EXPERIMENT/fairseq_params.txt
echo "MODEL_DIR: $MODELDIR" | tee -a $EVALDIR/$EXPERIMENT/fairseq_params.txt
echo "USER_DIR: $USERDIR" | tee -a $EVALDIR/$EXPERIMENT/fairseq_params.txt
echo "DATA_DIR: $DATADIR" | tee -a $EVALDIR/$EXPERIMENT/fairseq_params.txt
echo "BATCH_SIZE: $BATCH_SIZE" | tee -a $EVALDIR/$EXPERIMENT/fairseq_params.txt
echo "BEAM_PARAMS: $BEAM_PARAMS" | tee -a $EVALDIR/$EXPERIMENT/fairseq_params.txt


for MODEL in "${!MODELS[@]}"
do
    echo "$MODEL"

    OUTDIR=$EVALDIR/$EXPERIMENT/$MODEL
    CHECKDIR=$MODELDIR/$MODEL/checkpoint_best.pt

    mkdir -p $OUTDIR
    echo "CHECKPOINT: $CHECKDIR" | tee -a $OUTDIR/model_params.txt
    echo "MODEL_PARAMS: ${MODELS[$MODEL]}" | tee -a $OUTDIR/model_params.txt

    for SPLIT in test test1 test2 test3
    do

        fairseq-generate $DATADIR --path $CHECKDIR \
            --user-dir $USERDIR --task $TASK \
            --batch-size $BATCH_SIZE \
            $BEAM_PARAMS \
            --skip-invalid-size-inputs-valid-test \
            ${MODELS[$MODEL]} \
            --gen-subset ${SPLIT} \
            --fp16 \
            --memory-efficient-fp16 > $OUTDIR/output.${SPLIT}.txt


        grep ^H $OUTDIR/output.${SPLIT}.txt | cut -f3- > $OUTDIR/hypd.${SPLIT}.txt 
        grep ^T $OUTDIR/output.${SPLIT}.txt | cut -f2- > $OUTDIR/tard.${SPLIT}.txt 
        cat $OUTDIR/hypd.${SPLIT}.txt | sed 's/ ##//g' > $OUTDIR/hyp.${SPLIT}.txt
        cat $OUTDIR/tard.${SPLIT}.txt | sed 's/ ##//g' > $OUTDIR/tar.${SPLIT}.txt
        rm $OUTDIR/hypd.${SPLIT}.txt
        rm $OUTDIR/tard.${SPLIT}.txt
        files2rouge $OUTDIR/hyp.${SPLIT}.txt $OUTDIR/tar.${SPLIT}.txt | tee $OUTDIR/rouge.${SPLIT}.txt
    done
done
