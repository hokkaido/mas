ENC_OUTDIR=datasets/cnndm-augmented-511/preprocessed-core
ENT_OUTDIR=datasets/cnndm-augmented-511/preprocessed-entities
mkdir -p $ENC_OUTDIR
mkdir -p $ENT_OUTDIR
for SPLIT in train valid test; do 
    python src/preprocess_embeddings.py \
        --inputs datasets/outdated/cnndm/finished_files_510/${SPLIT}.abstract.txt \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.tgt \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.tgt \
        --workers 40; \
done 

for SPLIT in train valid test; do 
    python src/preprocess_embeddings.py \
        --inputs datasets/outdated/cnndm/finished_files_510/${SPLIT}.article.txt \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.src \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.src \
        --workers 40; \
done 