
import argparse
import contextlib
import sys

from collections import Counter
from multiprocessing import Pool

from pytorch_transformers import BertTokenizer

from .tokenizer import MagicBertTokenizer
from .ner import ENTITY_TYPES, align_tokens

import spacy

class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.max_tokens = getattr(args, 'max_len', None)

    def initializer(self):
        global bpe
        global nlp
        global mbpe
        nlp = spacy.load('en_core_web_sm')
        bpe = BertTokenizer.from_pretrained('bert-base-uncased')
        mbpe = MagicBertTokenizer.from_pretrained('bert-base-uncased')
    def encode(self, line):
        global bpe
        subword = bpe._tokenize(line)
        return subword

    def magic_encode(self, line):
        global mbpe
        subword = mbpe._tokenize(line)
        return subword

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_entities(self, line, subwords):
        global nlp
        doc = nlp(line)
       
        _, alignments = align_tokens(doc, subwords)

        entities = ['NONE' for subword in subwords]

        if alignments is None:
            return entities
            
        for i in range(len(alignments)):
            spacy_token = doc[i]
            ent_type = 'NONE'
            
            if spacy_token.ent_type_ in ENTITY_TYPES:
                ent_type = spacy_token.ent_type_

            for wp_idx in alignments[i]:
                entities[wp_idx] = ent_type
        
        return entities

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        ent_lines = []

        for line in lines:
            line = line.strip()
            if len(line) == 0:
                enc_lines.append('')
                ent_lines.append('')
            else:
                try:
                    tokens = self.encode(line)
                    mtokens = self.magic_encode(line)
                    entities = self.encode_entities(line, mtokens)

                    if self.max_tokens is not None:
                        tokens = tokens[:self.max_tokens]
                        entities = entities[:self.max_tokens]
                        
                    enc_lines.append(" ".join(tokens))
                    ent_lines.append(" ".join(entities))
                except AssertionError:
                    print('ERROR')

        assert len(enc_lines) == len(ent_lines)
        return ["PASS", enc_lines, ent_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]
