import argparse

from mas import SentenceSelector


def main():
    spacy.require_gpu()

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
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1000)

    args = parser.parse_args()

    selector = SentenceSelector(args.k)

    i = 0
    with open(args.article_path) as article_file, open(args.abstract_path) as abstract_file, open(args.output_path, 'w+') as output_file:
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
                    print(selector.confine_docs(articles[j], abstracts[j]), file=output_file)
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
                print(selector.confine_docs(articles[j], abstracts[j]), file=output_file)

if __name__ == "__main__":
    main()
