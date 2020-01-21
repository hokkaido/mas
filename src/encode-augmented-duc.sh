ENC_OUTDIR=datasets/duc2004/preprocessed-core
ENT_OUTDIR=datasets/duc2004/preprocessed-entities
mkdir -p $ENC_OUTDIR
mkdir -p $ENT_OUTDIR
for SPLIT in test; do 
    python src/preprocess_embeddings.py \
        --inputs datasets/duc2004/preprocessed/${SPLIT}.src \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.src \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.src \
        --workers 40; \
done 

for BATCH in 1 2 3 4; do 
    python src/preprocess_embeddings.py \
        --inputs datasets/duc2004/preprocessed/test.${BATCH}.tgt \
        --enc-outputs ${ENC_OUTDIR}/test.${BATCH}.tgt \
        --ent-outputs ${ENT_OUTDIR}/test.${BATCH}.tgt \
        --workers 40; \
done 