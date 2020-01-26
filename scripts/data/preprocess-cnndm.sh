DATA_DIR=datasets

ENC_OUTDIR=${DATA_DIR}/cnndm/preprocessed-core
ENT_OUTDIR=${DATA_DIR}/cnndm/preprocessed-entities

mkdir -p $ENC_OUTDIR
mkdir -p $ENT_OUTDIR

for SPLIT in train valid test; do 
    python preprocess.py \
        --inputs ${DATA_DIR}/cnndm/preprocessed/${SPLIT}.abstract.txt \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.tgt \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.tgt \
        --workers 40; \
done 

for SPLIT in train valid test; do 
    python preprocess.py \
        --inputs ${DATA_DIR}/cnndm/preprocessed/${SPLIT}.article.txt \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.src \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.src \
        --workers 40; \
done 