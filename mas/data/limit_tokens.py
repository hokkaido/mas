
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
        help="input files to limit",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save limited outputs",
    )
    parser.add_argument("--max_tokens", type=int, default=510)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8"))
            if input != "-" else sys.stdin
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.limit_lines, zip(*inputs), 100)

        stats = Counter()
        for i, (filt, limited_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for limited_line, output_h in zip(limited_lines, outputs):
                    print(limited_line, file=output_h)
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
        pass

    def limit_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        limited_lines = []

        for line in lines:
            line = line.strip()
            line = ' '.join(line.split(' ')[:self.args.max_tokens])
            limited_lines.append(line)
        return ["PASS", limited_lines]

if __name__ == "__main__":
    main()
