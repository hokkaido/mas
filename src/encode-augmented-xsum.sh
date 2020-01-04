ENC_OUTDIR=datasets/xsum/preprocessed-core
ENT_OUTDIR=datasets/xsum/preprocessed-entities
mkdir -p $ENC_OUTDIR
mkdir -p $ENT_OUTDIR


for SPLIT in validation test train; do 
    python src/preprocess_embeddings.py \
        --inputs datasets/xsum/preprocessed/${SPLIT}.src \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.src \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.src \
        --workers 20; \
done 