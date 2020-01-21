import time
import argparse
import contextlib
import sys
from collections import Counter, namedtuple
import spacy
from heapq import heappush, nlargest
import math
import csv
RankedSentence = namedtuple('RankedSentence', ['rank', 'position', 'text'])

def get_sent_pos(doc):
    lookup = {}
    for sent_i, sent in enumerate(doc.sents):
        lookup[sent.start] = sent_i
        lookup[sent.end] = sent_i

    return lookup

def entity_counts(doc):
    c = Counter()
    for ent in doc.ents:
        c[ent.text] += 1
        
    return c

def rank_sentences(article_sents, abstract_doc):   
    abstract_count = entity_counts(abstract_doc)
    ranked = []
    for sent in article_sents:
        rank = 0
        for ent in sent.ents:
            if ent.text in abstract_count:
                rank += abstract_count[ent.text]
            else:
                rank += 0.5
        
        heappush(ranked, RankedSentence(rank, sent.doc._.sent_pos[sent.start], sent.text))
        
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



def extract_ranked_sentences(article, abstract, k=16):
    start, end = lcs(article, abstract)
    start_sent = article._.sent_pos[start]
    end_sent = article._.sent_pos[end]
    constrained = list(article.sents)[start_sent:end_sent]
    ranked_sentences = nlargest(k, rank_sentences(constrained, abstract))
    return sorted(ranked_sentences, key=lambda x: x[1])

class SentenceSelector:
    def __init__(self, k):
        self.nlp = spacy.load('en_core_web_sm')
        self.k = k

    def confine_texts(self, article, abstract):
        return self.confine_docs(self.nlp(article), self.nlp(abstract))

    def confine_docs(self, article, abstract):
        if len(article) == 0 or len(abstract) == 0:
            return '', ''
        ranked_sentences = extract_ranked_sentences(article, abstract, self.k)
        ranked_pos = [s.position for s in ranked_sentences]
        article_sents = list(article.sents)
        labels = []

        for i, s in enumerate(article_sents):
            label = '__label__0'
            group = []
            if i > 0:
                group.append(article_sents[i - 1].text)
            else:
                group.append('[SC]')
            group.append(s.text)
            if i < len(article_sents) - 1:
                group.append(article_sents[i + 1].text)
            else:
                group.append('[EC]')
                
            if i in ranked_pos:
                label = '__label__1'
            
            labels.append([label, ' '.join(group)])

        return labels

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
        help="path to save pos sents",
    )
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1000)

    args = parser.parse_args()

    selector = SentenceSelector(args.k)

    i = 0
    with open(args.article_path) as article_file, open(args.abstract_path) as abstract_file, open(args.output_path,'w+') as output_file:
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
                    labels = selector.confine_docs(articles[j], abstracts[j])
                    for label in labels:
                        print(' '.join(label), file=output_file)
 
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
                labels = selector.confine_docs(articles[j], abstracts[j])
                for label in labels:
                    print(' '.join(label), file=output_file)

if __name__ == "__main__":
    spacy.tokens.Doc.set_extension("sent_pos", getter=get_sent_pos)

    main()
