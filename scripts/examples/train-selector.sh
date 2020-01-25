

LABEL_DIR=datasets/constrained_classification/k16

python train_selector.py  --train-path ${LABEL_DIR}/fasttext.train.txt --eval-path ${LABEL_DIR}/fasttext.valid.txt --sample 0.01