DATA_DIR=datasets
DATA_DIR=datasets

ENC_OUTDIR=${DATA_DIR}/duc2004/preprocessed-core
ENT_OUTDIR=${DATA_DIR}/duc2004/preprocessed-entities
mkdir -p $ENC_OUTDIR
mkdir -p $ENT_OUTDIR
for SPLIT in test; do 
    python preprocess.py \
        --inputs ${DATA_DIR}/duc2004/preprocessed/${SPLIT}.src \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.src \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.src \
        --max-len 511 \
        --workers 40; \
done 

for BATCH in 1 2 3 4; do 
    python preprocess.py \
        --inputs ${DATA_DIR}/duc2004/preprocessed/test.${BATCH}.tgt \
        --enc-outputs ${ENC_OUTDIR}/test.${BATCH}.tgt \
        --ent-outputs ${ENT_OUTDIR}/test.${BATCH}.tgt \
        --max-len 511 \
        --workers 40; \
done 