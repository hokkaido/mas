ENC_OUTDIR=datasets/cnndm-entities/preprocessed
ENT_OUTDIR=datasets/cnndm-entities/entities
mkdir -p $ENC_OUTDIR
mkdir -p $ENT_OUTDIR
for SPLIT in train valid test; do 
    python src/preprocess_embeddings.py \
        --inputs datasets/outdated/cnndm/finished_files_510/${SPLIT}.abstract.txt \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.tgt \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.tgt \
        --workers 30; \
done 

for SPLIT in train valid test; do 
    python src/preprocess_embeddings.py \
        --inputs datasets/outdated/cnndm/finished_files_510/${SPLIT}.article.txt \
        --enc-outputs ${ENC_OUTDIR}/${SPLIT}.src \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.src \
        --workers 30; \
done 