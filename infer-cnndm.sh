MODEL=checkpoints/checkpoint_best.pt
DATADIR=datasets/cnndmsent/processed
USERDIR=deps/MASS/MASS-summarization/mass

fairseq-generate $DATADIR --path $MODEL \
    --user-dir $USERDIR --task summarization_mass \
    --batch-size 64 --beam 5 --min-len 50 --no-repeat-ngram-size 3 \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --memory-efficient-fp16 \
    --lenpen 1.0 \


    #grep ^H eval/cnndm/output.txt | cut -f3- > hyp3.txt 
    #grep ^T eval/cnndm/output.txt | cut -f2- > tar3.txt 
    #(mas) goto@goto:~/dev/mas$ cat hyp3.txt | sed 's/ ##//g' > hyp4.txt
    #(mas) goto@goto:~/dev/mas$ cat tar3.txt | sed 's/ ##//g' > tar4.txt
