import argparse
import os
from lxml import etree
from os.path import join

class DUC2004SourceReader:
    def __init__(self, source_path):
        self.docs = {}
        self.load(source_path)
        print(len(self.docs))

    def load(self, source_path):
        parser = etree.XMLParser(remove_blank_text=True)
        for topic_dir in os.listdir(source_path):
            for doc_file in os.listdir(os.path.join(source_path, topic_dir)):
                doc_path = os.path.join(source_path, topic_dir, doc_file)
                with open(doc_path, 'r') as f:
                    doc = f.read()
                    doc = doc.replace('&AMP;', '&amp;')
                
                    doc = etree.XML(doc, parser)
                    self.docs[topic_dir + '.' + doc_file] = ' '.join([line.strip() for line in doc.find('TEXT').text.lstrip().splitlines()])

class DUC2004TargetReader:
    def __init__(self, target_path):
        self.docs = {}
        self.load(target_path)
        print(len(self.docs))

    def load(self, target_path):
        for target_file in os.listdir(target_path):
            with open(os.path.join(target_path, target_file), 'r') as f:
                self.docs[target_file] = f.read()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default='datasets/duc2004/unprocessed/docs',
    )
    parser.add_argument(
        "--target",
        default='datasets/duc2004/unprocessed/eval/models/1',
    )
    parser.add_argument(
        "--output",
        default='datasets/duc2004/preprocessed/',
    )
    args = parser.parse_args()

    source_reader = DUC2004SourceReader(args.source)
    target_reader = DUC2004TargetReader(args.target)
    #targets = load_tgt(args.tgt_path)

    with open(join(args.output,'test.src'), 'w+') as sf, \
        open(join(args.output,'test.1.tgt'), 'w+') as t1, \
        open(join(args.output,'test.2.tgt'), 'w+') as t2, \
        open(join(args.output,'test.3.tgt'), 'w+') as t3, \
        open(join(args.output,'test.4.tgt'), 'w+') as t4:
        t_out = [t1, t2, t3, t4]
        sf.write('\n'.join(source_reader.docs.values()))
        for source, source_doc in source_reader.docs.items():
            print(source)
            count = 0
            for target, target_doc in target_reader.docs.items():
                segments = target.split('.')
                cluster = '{}t.{}.{}'.format(segments[0].lower(), segments[-2], segments[-1])
                if cluster in source:
                    t_out[count].write(target_doc.strip() + '\n')
                    count += 1
                #print(target.split('.'))
    



if __name__ == "__main__":
    main()