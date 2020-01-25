#!/bin/bash

EXPERIMENT=exp2-xsum
TASK=augmented_summarization_mass
EVALDIR=eval
MODELDIR=checkpoints
DATADIR=datasets/xsum
USERDIR=deps/MASS/MASS-summarization/mass
BATCH_SIZE=64
BEAM_PARAMS="--beam 4 --min-len 5 --no-repeat-ngram-size 3 --max-len-b 50 --lenpen 1.9"

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

    fairseq-generate $DATADIR --path $CHECKDIR \
        --user-dir $USERDIR --task $TASK \
        --batch-size $BATCH_SIZE \
        $BEAM_PARAMS \
        --skip-invalid-size-inputs-valid-test \
        ${MODELS[$MODEL]} \
        --fp16 \
        --memory-efficient-fp16 > $OUTDIR/output.txt


    grep ^H $OUTDIR/output.txt | cut -f3- > $OUTDIR/hypd.txt 
    grep ^T $OUTDIR/output.txt | cut -f2- > $OUTDIR/tard.txt 
    cat $OUTDIR/hypd.txt | sed 's/ ##//g' > $OUTDIR/hyp.txt
    cat $OUTDIR/tard.txt | sed 's/ ##//g' > $OUTDIR/tar.txt
    rm $OUTDIR/hypd.txt
    rm $OUTDIR/tard.txt
    files2rouge $OUTDIR/hyp.txt $OUTDIR/tar.txt | tee $OUTDIR/rouge.txt
done
