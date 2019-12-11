ENC_OUTDIR=datasets/cnndm-constrained-510/k16-generated/preprocessed-core
ENT_OUTDIR=datasets/cnndm-constrained-510/k16-generated/preprocessed-entities
mkdir -p $ENC_OUTDIR
mkdir -p $ENT_OUTDIR

for SPLIT in test; do 
    python src/preprocess_embeddings.py \
        --inputs datasets/constrained_classification/k16/test/${SPLIT}.article.txt \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.src \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.src \
        --workers 40; \
done