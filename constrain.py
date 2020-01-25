import argparse
import spacy
import time
from mas import SentenceSelector
import sys
from contextlib import ExitStack

def write_classifier(out_file, ranked, unranked):
    for s in ranked:
        print('__label__1 {}'.format(s.text), file=out_file)
    for s in unranked:
        print('__label__0 {}'.format(s.text), file=out_file)

def write_constrained(out_file, ranked):
    print(' '.join([s.text for s in ranked]), file=out_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--article-path",
        help="path to articles to constrain",
    )
    parser.add_argument(
        "--abstract-path",
        help="abstracts to constrain",
    )
    parser.add_argument(
        "--output-path",
        help="path to save constrained articles",
    )
    parser.add_argument(
        "--classifier-output-path",
        help="path to save constrained classifier samples",
    )
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1000)

    args = parser.parse_args()

    selector = SentenceSelector(args.k)

    i = 0

    with ExitStack() as stack:

        article_file = stack.enter_context(open(args.article_path))
        abstract_file = stack.enter_context(open(args.abstract_path))
        output_file = stack.enter_context(open(args.output_path, 'w+'))
        classifier_out_file = None
        if getattr(args, 'classifier_output_path', False):
            classifier_out_file = stack.enter_context(open(args.classifier_output_path, 'w+'))

        start = time.time()
        articles = []
        abstracts = []
        for x, y in zip(article_file, abstract_file):

            x = x.strip()
            y = y.strip()
            articles.append(x)
            abstracts.append(y)
            
            if i % args.batch_size == 0:
                articles = list(selector.nlp.pipe(articles))
                abstracts = list(selector.nlp.pipe(abstracts))
                for j in range(len(articles)):
                    ranked, unranked = selector.confine_docs(articles[j], abstracts[j])
                    write_constrained(output_file, ranked)
                    if classifier_out_file is not None:
                        write_classifier(classifier_out_file, ranked, unranked)

                articles = []
                abstracts = []
                end = time.time()
                print('elapsed since start', end - start)
                print("processed {} lines".format(i), file=sys.stderr)
                
            i += 1
        if len(articles) > 0:
            print('leftovers')
            articles = list(selector.nlp.pipe(articles))
            abstracts = list(selector.nlp.pipe(abstracts))
            for j in range(len(articles)):
                ranked, unranked = selector.confine_docs(articles[j], abstracts[j])
                write_constrained(output_file, ranked)
                if classifier_out_file is not None:
                    write_classifier(classifier_out_file, ranked, unranked)

if __name__ == "__main__":
    main()
