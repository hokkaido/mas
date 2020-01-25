from itertools import product
from os import path

datasets = ['preprocessed-core', 'preprocessed-entities']
splits = ['test', 'train', 'valid']
langs = ['src', 'tgt']

DATA_PATH = 'datasets/gigaword-augmented'

def main():
    for split in splits:
        for lang in langs:
            with open(path.join(DATA_PATH, 'preprocessed-entities/{}.{}'.format(split, lang))) as f:
                print('reading core/{}.{}'.format(split, lang))
                line = f.readline().rstrip()
                cnt = 0
                while line:
                    if len(line) < 5:
                        print(line, 'broken line at ', cnt)
                    line = f.readline().rstrip()
                    cnt += 1
        

if __name__ == "__main__":
    main()