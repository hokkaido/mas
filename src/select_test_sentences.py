import spacy
import argparse
import transformers
from classification_model import ClassificationModel

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
    model = ClassificationModel('bert', 'outputs/checkpoint-25000', num_labels=2)
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