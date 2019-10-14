OUTDIR=datasets/cnndmsent/para-400

mkdir -p $OUTDIR

for SPLIT in train valid test; do 
    python deps/MASS/MASS-summarization/encode.py \
        --inputs datasets/outdated/cnndm/finished_files_new/${SPLIT}.abstract.txt \
        --outputs ${OUTDIR}/${SPLIT}.tgt \
        --workers 60; \
done 

for SPLIT in train valid test; do 
    python deps/MASS/MASS-summarization/encode.py \
        --inputs datasets/outdated/cnndm/finished_files_new/${SPLIT}.article.txt \
        --outputs ${OUTDIR}/${SPLIT}.src \
        --workers 60; \
done 