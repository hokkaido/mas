import spacy
import argparse
import transformers
from .classification_model import ClassificationModel
import pandas as pd
from sklearn.utils import class_weight

def constrain_sentences(sents, preds, k=16):
    assert len(sents) == len(preds)

    selected = []
    for i in range(len(sents)):
        if preds[i]:
            selected.append(sents[i])
        if len(selected) == k:
            return selected

    return selected
    
def constrain_with_transformer(args):
    nlp = spacy.load('en_core_web_sm')
    model = ClassificationModel(args.model, args.model_dir, num_labels=2)
    i = 0
    with open(args.article_path) as article_file, open(args.output_path, 'w+') as output_file:
        
            
        for x in article_file:
            if i % 500 == 0:
                print('line ', i)
            x = x.strip()
            sents = [sent.text for sent in nlp(x).sents]
            predictions, raw_outputs = model.predict(sents)
            constrained = constrain_sentences(sents, predictions)
            print(' '.join(constrained), file=output_file)

            i += 1