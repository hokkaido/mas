ENC_OUTDIR=datasets/gigaword-augmented/preprocessed-core
ENT_OUTDIR=datasets/gigaword-augmented/preprocessed-entities
mkdir -p $ENC_OUTDIR
mkdir -p $ENT_OUTDIR
for SPLIT in test valid train; do 
    python src/preprocess_embeddings.py \
        --inputs datasets/gigaword-augmented/cleaned/${SPLIT}.abstract.txt \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.tgt \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.tgt \
        --workers 15; \
done 

for SPLIT in test valid train; do 
    python src/preprocess_embeddings.py \
        --inputs datasets/gigaword-augmented/cleaned/${SPLIT}.article.txt \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.src \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.src \
        --workers 15; \
done 