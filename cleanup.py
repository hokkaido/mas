import argparse

from mas import CnnDmWriter, XSumWriter, DUC2004Writer

DATA_WRITERS = {
    'cnndm': CnnDmWriter,
    'xsum': XSumWriter,
    'duc2004': DUC2004Writer
}

def create_writer(args):
    return DATA_WRITERS[args.config](args)

def main(args):
    writer = create_writer(args)
    writer.write()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, choices=['cnndm', 'xsum', 'duc2004'], required=True)
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--max-len', type=int)

    args = parser.parse_args()
    main(args)