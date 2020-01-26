import spacy
import argparse
import transformers
from mas import constrain_with_transformer

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
        type=str,
        required=True,
        help="path to articles to constrain",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="path to save constrained articles",
    )
    parser.add_argument(
        "--model-dir",
        help="path to model checkpoint",
        required=True
    )
    parser.add_argument(
        "--model",
        help="model name, default xlnet",
        default='xlnet',
        required=True
    )
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1000)

    args = parser.parse_args()

    constrain_with_transformer(args)

            
if __name__ == "__main__":
    main()