DATA_DIR=datasets

ENC_OUTDIR=${DATA_DIR}/xsum/preprocessed-core
ENT_OUTDIR=${DATA_DIR}/xsum/preprocessed-entities
mkdir -p $ENC_OUTDIR
mkdir -p $ENT_OUTDIR

for SPLIT in valid test train; do 
    python preprocess.py \
        --inputs ${DATA_DIR}/xsum/preprocessed/${SPLIT}.tgt \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.tgt \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.tgt \
        --max-len 511 \
        --workers 20; \
done 

for SPLIT in valid test train; do 
    python preprocess.py \
        --inputs ${DATA_DIR}/xsum/preprocessed/${SPLIT}.src \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.src \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.src \
        --max-len 511 \
        --workers 20; \
done