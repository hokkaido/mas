#!/bin/bash
EVALDIR=eval
EXPERIMENT=exp4-duc-xsum

declare -A MODELS

MODELS=(
    ['xsum-vanilla']=''
    ['xsum-entities-encoder']="--embed-entities-encoder"
    ['xsum-entities-encoder-segments-encoder']="--embed-entities-encoder --embed-segments-encoder --segment-tokens . --max-segments 128"
    ['xsum-segments-encoder']="--embed-segments-encoder --segment-tokens . --max-segments 128"
)

for MODEL in "${!MODELS[@]}"
do
    echo "$MODEL"

    OUTDIR=$EVALDIR/$EXPERIMENT/$MODEL



    cat $OUTDIR/tar.test*.txt >> $OUTDIR/tar.txt
    cat $OUTDIR/hyp.test*.txt >> $OUTDIR/hyp.txt

    files2rouge $OUTDIR/hyp.txt $OUTDIR/tar.txt -a "-c 95 -b 75 -m -n 4 -w 1.2 -a" | tee $OUTDIR/rouge.txt


done
