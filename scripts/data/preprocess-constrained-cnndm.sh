DATA_DIR=datasets

ENC_OUTDIR=${DATA_DIR}/cnndm-constrained/preprocessed-core
ENT_OUTDIR=${DATA_DIR}/cnndm-constrained/preprocessed-entities

mkdir -p $ENC_OUTDIR
mkdir -p $ENT_OUTDIR

for SPLIT in test; do 
    python preprocess.py \
        --inputs ${DATA_DIR}/cnndm-constrained/preprocessed/${SPLIT}.tgt \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.tgt \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.tgt \
        --workers 40; \
done 

for SPLIT in test; do 
    python preprocess.py \
        --inputs ${DATA_DIR}/cnndm-constrained/preprocessed/${SPLIT}.src \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.src \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.src \
        --workers 40; \
done 