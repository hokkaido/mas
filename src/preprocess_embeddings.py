
import argparse
import contextlib
import sys

from collections import Counter
from multiprocessing import Pool

from pytorch_transformers import BertTokenizer

from tokenizer import MagicBertTokenizer
from ner import ENTITY_TYPES, align_tokens

import spacy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--enc-outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--ent-outputs",
        nargs="+",
        default=['-'],
        help="path to save entities outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.enc_outputs), \
        "number of input and output paths should match"

    assert len(args.inputs) == len(args.ent_outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.inputs
        ]
        enc_outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.enc_outputs
        ]
        ent_outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.ent_outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, enc_lines, ent_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, enc_outputs):
                    print(enc_line, file=output_h)
                for ent_line, output_h in zip(ent_lines, ent_outputs):
                    print(ent_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)

class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

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
            print(subwords)
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
            if len(line) == 0 and not self.args.keep_empty:
                return ["EMPTY", None, None]
            try:
                tokens = self.encode(line)
                mtokens = self.magic_encode(line)
                entities = self.encode_entities(line, mtokens)

                enc_lines.append(" ".join(tokens))
                ent_lines.append(" ".join(entities))
            except AssertionError:
                print('Error occured')
        return ["PASS", enc_lines, ent_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
