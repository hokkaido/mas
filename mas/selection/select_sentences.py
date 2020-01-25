import time
import argparse
import contextlib
import sys
from collections import Counter, namedtuple
import spacy
from heapq import heappush, nlargest
import math

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


spacy.tokens.Doc.set_extension("sent_pos", getter=get_sent_pos)
