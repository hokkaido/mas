
import argparse
import contextlib
import sys

from collections import Counter
from multiprocessing import Pool

from pytorch_transformers import BertTokenizer

from mas import MultiprocessingEncoder

import spacy

def main(args):
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
        encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 1000)

        stats = Counter()
        for i, (filt, enc_lines, ent_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, enc_outputs):
                    print(enc_line, file=output_h)
                for ent_line, output_h in zip(ent_lines, ent_outputs):
                    print(ent_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 1000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)



if __name__ == "__main__":
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
    parser.add_argument(
        "--max-len",
        type=int,
        help="whether to truncate the output",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    main(args)
