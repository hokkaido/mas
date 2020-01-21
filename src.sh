grep ^S output.txt | cut -f2- > srcd.txt 
cat srcd.txt | sed 's/ ##//g' > src.txt
