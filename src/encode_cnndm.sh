mkdir -p datasets/cnndmsent/para
for SPLIT in train valid test; do 
    python deps/MASS/MASS-summarization/encode.py \
        --inputs datasets/outdated/cnndm/finished_files_new/${SPLIT}.abstract.txt \
        --outputs datasets/cnndmsent/para/${SPLIT}.tgt.txt \
        --workers 60; \
done 

for SPLIT in train valid test; do 
    python deps/MASS/MASS-summarization/encode.py \
        --inputs datasets/outdated/cnndm/finished_files_new/${SPLIT}.article.txt \
        --outputs datasets/cnndmsent/para/${SPLIT}.src.txt \
        --workers 60; \
done 