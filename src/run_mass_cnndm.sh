save_dir=checkpoints/cnndm
user_dir=deps/MASS/MASS-fairseq/mass
data_dir=datasets/cnndm/fqprocessed

mkdir -p $save_dir

fairseq-train $data_dir \
    --user-dir $user_dir \
    --save-dir $save_dir \
    --task xmasked_seq2seq \
    --source-langs ar,ab \
    --target-langs ar,ab \
    --langs ar,ab \
    --arch xtransformer \
    --memt_steps ar-ab \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --lr 0.0001 --min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy \
    --dropout 0.1 --relu-dropout 0.1 --attention-dropout 0.1 \
    --max-update 300000 \
    --share-decoder-input-output-embed \
    --valid-lang-pairs ar-ab \
    --word_mask 0.15 \
    --fp16 \
    --update-freq 8 \