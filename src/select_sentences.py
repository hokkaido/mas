import time
import argparse
import contextlib
import sys
from collections import Counter, namedtuple
import spacy
from heapq import heappush, nlargest
import math

RankedSentence = namedtuple('RankedSentence', ['rank', 'position', 'text'])

def entity_counts(doc):
    c = Counter()
    for ent in doc.ents:
        c[ent.text] += 1
        
    return c

def rank_sentences(article_doc, abstract_doc):   
    abstract_count = entity_counts(abstract_doc)
    ranked = []
    i = 0
    for sent in article_doc.sents:
        rank = 0
        for ent in sent.ents:
            if ent.text in abstract_count:
                rank += abstract_count[ent.text]
            else:
                rank += 0.5
        
        heappush(ranked, RankedSentence(rank, i, sent.text))
        i += 1
        
    return ranked

def lcs(a, b):
    # generate matrix of length of longest common subsequence for substrings of both words
    lengths = [[0] * (len(b)+1) for _ in range(len(a)+1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x.text.lower() == y.text.lower():
                lengths[i+1][j+1] = lengths[i][j] + 1
            else:
                lengths[i+1][j+1] = max(lengths[i+1][j], lengths[i][j+1])
 
    # read a substring from the matrix
    start = math.inf
    end = -math.inf
    
    j = len(b)
    for i in range(1, len(a)+1):
        if lengths[i][j] != lengths[i-1][j]:
            start = min(a[i-1].sent.start, start)
            end = max(a[i-1].sent.end, end)
 
    return start, end



def extract_ranked_sentences(article, abstract, k=8):
    start, end = lcs(article, abstract)
    ranked_sentences = nlargest(k, rank_sentences(article[start:end].as_doc(), abstract))
    return sorted(ranked_sentences, key=lambda x: x[1])

class SentenceSelector:
    def __init__(self, k):
        self.nlp = spacy.load('en_core_web_sm')
        self.k = k

    def confine_texts(self, article, abstract):
        return self.confine_docs(self.nlp(article), self.nlp(abstract))

    def confine_docs(self, article, abstract):
        if len(article) == 0:
            return ''
        ranked_sentences = extract_ranked_sentences(article, abstract, self.k)
        return ' '.join(s.text for s in ranked_sentences)

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
