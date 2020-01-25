import spacy
import argparse
import transformers
from .classification_model import ClassificationModel
import pandas as pd
from sklearn.utils import class_weight

def load_df(pos_fp, neg_fp, frac=1):
    df_pos = pd.DataFrame(open(pos_fp, 'r').read().split('\n'), columns=['text'])
    df_pos['label'] = 1
    df_neg = pd.DataFrame(open(neg_fp, 'r').read().split('\n'), columns=['text'])
    df_neg['label'] = 0
    return pd.concat([df_pos, df_neg]).sample(frac=frac).reset_index(drop=True)


def constrain_sentences(sents, preds, k=16):
    assert len(sents) == len(preds)

    selected = []
    for i in range(len(sents)):
        if preds[i]:
            selected.append(sents[i])
        if len(selected) == k:
            return selected

    return selected
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--article-path",
        help="path to articles to constrain",
    )
    parser.add_argument(
        "--output-path",
        help="path to save constrained articles",
    )
    
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1000)

    args = parser.parse_args()

    nlp = spacy.load('en_core_web_sm')
    model = ClassificationModel('xlnet', 'outputs/checkpoint-8000', num_labels=2)
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
if __name__ == "__main__":
    main()