ENC_OUTDIR=datasets/cnndm-constrained-510/k16/preprocessed-core
ENT_OUTDIR=datasets/cnndm-constrained-510/k16/preprocessed-entities
mkdir -p $ENC_OUTDIR
mkdir -p $ENT_OUTDIR

for SPLIT in valid test; do 
    python src/preprocess_embeddings.py \
        --inputs datasets/outdated/cnndm/constrained_files/k16/${SPLIT}.article.txt \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.src \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.src \
        --workers 40; \
done 