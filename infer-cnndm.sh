MODEL=checkpoints/checkpoint_best.pt
DATADIR=datasets/cnndm
USERDIR=deps/MASS/MASS-summarization/mass
RESDIR=eval/cnndm
fairseq-generate $DATADIR --path $MODEL \
    --user-dir $USERDIR --task translation_mass \
    --batch-size 64 --beam 5 --min-len 50 --no-repeat-ngram-size 3 \
    --lenpen 1.0 \


    #grep ^H eval/cnndm/output.txt | cut -f3- > hyp.txt 
    #grep ^T eval/cnndm/output.txt | cut -f2- > tar.txt 
    #(mas) goto@goto:~/dev/mas$ cat hyp.txt | sed 's/ ##//g' > hyp2.txt
    #(mas) goto@goto:~/dev/mas$ cat tar.txt | sed 's/ ##//g' > tar2.txt
