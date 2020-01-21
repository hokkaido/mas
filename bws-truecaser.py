from sacremoses import MosesTruecaser, MosesTokenizer

mtr = MosesTruecaser()
mtok = MosesTokenizer(lang='en')

# Save the truecase model to 'big.truecasemodel' using `save_to`
tokenized_docs = [mtok.tokenize(line) for line in open('fi')]
mtr.train(tokenized_docs, save_to='cnndm.truecasemodel')