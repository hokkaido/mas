grep ^H output.txt | cut -f3- > hypd.txt 
grep ^T output.txt | cut -f2- > tard.txt 
cat hypd.txt | sed 's/ ##//g' > hyp.txt
cat tard.txt | sed 's/ ##//g' > tar.txt
rm hypd.txt
rm tard.txt
files2rouge hyp.txt tar.txt