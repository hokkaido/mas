import spacy
import argparse
import transformers
from classification_model import ClassificationModel
import pandas as pd
from sklearn.utils import class_weight
import fasttext
from heapq import heappush, nlargest
from collections import Counter, namedtuple

RankedSentence = namedtuple('RankedSentence', ['rank', 'position', 'text'])

def kadane(arr):
    if not arr:
        return []  # corner case
    maxi = wini = maxj = 0
    maxseen = 0
    winmax = 0  # max sum of window ends at winj
    for winj in range(len(arr)):
        if winmax + arr[winj].rank < arr[winj].rank:
            winmax = arr[winj].rank
            wini = winj
        else:
            winmax += arr[winj].rank
        if winmax > maxseen:
            maxi = wini
            maxj = winj
    return arr[maxi:maxj+1]

def constrain_sentences(sents, preds, probs, k=16):
    assert len(sents) == len(preds)

    ranks = []
    for i in range(len(sents)):
        if preds[i][0] == '__label__1':
            heappush(ranks, RankedSentence(probs[i][0], i, sents[i]))
        else:
            heappush(ranks, RankedSentence(probs[i][1], i, sents[i]))
        
    selected = nlargest(k, ranks)

    return sorted(selected, key=lambda x: x[1])
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--article-path",
        help="path to articles to constrain",
    )
    parser.add_argument(
        "--model-path",
        help="path to fasttext model",
    )
    parser.add_argument(
        "--output-path",
        help="path to save constrained articles",
    )
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1000)

    args = parser.parse_args()

    nlp = spacy.load('en_core_web_sm')
    model = model = fasttext.load_model(args.model_path)
    i = 0

    with open(args.article_path) as article_file, open(args.output_path, 'w+') as output_file:
        
            
        for x in article_file:
            
            if i % 500 == 0:
                print('line ', i)
            x = x.strip()
            sents = [sent.text for sent in nlp(x).sents]
            predictions, probs = model.predict(sents, k=2)
           
            constrained = constrain_sentences(sents, predictions, probs)
            
            print(' '.join([r.text for r in constrained]), file=output_file)

            i += 1
if __name__ == "__main__":
    main()