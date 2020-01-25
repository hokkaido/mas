#!/bin/bash

EXPERIMENT=exp10-duc-cnndm
TASK=augmented_summarization_mass
EVALDIR=eval
MODELDIR=checkpoints
DATADIR=datasets/duc2004
USERDIR=deps/MASS/MASS-summarization/mass
BATCH_SIZE=64
BEAM_PARAMS="--beam 5 --min-len 45 --no-repeat-ngram-size 4 --max-len-b 183 --lenpen 2.0"

declare -A MODELS

MODELS=(
    ['cnndm-entities-encoder']="--embed-entities-encoder"
)

# Save fairseq params for experiment
echo "TASK: $TASK" | tee $EVALDIR/$EXPERIMENT/fairseq_params.txt
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
    
    echo "CHECKPOINT: $CHECKDIR" | tee $OUTDIR/model_params.txt
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
            --cpu > $OUTDIR/output.${SPLIT}.txt


        grep ^H $OUTDIR/output.${SPLIT}.txt | cut -f3- > $OUTDIR/hypd.${SPLIT}.txt 
        grep ^T $OUTDIR/output.${SPLIT}.txt | cut -f2- > $OUTDIR/tard.${SPLIT}.txt 
        cat $OUTDIR/hypd.${SPLIT}.txt | sed 's/ ##//g' >> $OUTDIR/hyp.txt
        cat $OUTDIR/tard.${SPLIT}.txt | sed 's/ ##//g' >> $OUTDIR/tar.txt
        rm $OUTDIR/hypd.${SPLIT}.txt
        rm $OUTDIR/tard.${SPLIT}.txt
        
    done
    files2rouge $OUTDIR/hyp.txt $OUTDIR/tar.txt -a "-c 95 -b 75 -m -n 4 -w 1.2 -a" | tee $OUTDIR/rouge.txt
done
