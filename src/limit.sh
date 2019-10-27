ENC_OUTDIR=datasets/cnndm-entities/preprocessed-510
ENT_OUTDIR=datasets/cnndm-entities/entities-510
mkdir -p $ENC_OUTDIR
mkdir -p $ENT_OUTDIR

for SPLIT in train valid test; do 
    python src/limit_tokens.py \
        --inputs datasets/cnndm-entities/preprocessed/${SPLIT}.tgt \
        --outputs ${ENC_OUTDIR}/${SPLIT}.tgt \
        --workers 30; \
done 

for SPLIT in train valid test; do 
    python src/limit_tokens.py \
        --inputs datasets/cnndm-entities/preprocessed/${SPLIT}.src \
        --outputs ${ENC_OUTDIR}/${SPLIT}.src \
        --workers 30; \
done 

for SPLIT in train valid test; do 
    python src/limit_tokens.py \
        --inputs datasets/cnndm-entities/entities/${SPLIT}.tgt \
        --outputs ${ENT_OUTDIR}/${SPLIT}.tgt \
        --workers 30; \
done 

for SPLIT in train valid test; do 
    python src/limit_tokens.py \
        --inputs datasets/cnndm-entities/entities/${SPLIT}.src \
        --outputs ${ENT_OUTDIR}/${SPLIT}.src \
        --workers 30; \
done 