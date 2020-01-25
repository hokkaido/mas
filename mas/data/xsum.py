import argparse 
import json
import os

def process_line(doc):
    allcovered = 0
    modeFlag = None
    docdata = []
    summarydata = []

    for sent in doc:
        if "[SN]URL[SN]" in sent:
            modeFlag = "URL"
            allcovered += 1
        elif "[SN]FIRST-SENTENCE[SN]" in sent:
            modeFlag = "INTRODUCTION"
            allcovered += 1
        elif "[SN]RESTBODY[SN]" in sent:
            modeFlag = "RestBody"
            allcovered += 1
        else:
            if modeFlag == "RestBody":
                docdata.append(sent)
            if modeFlag == "INTRODUCTION":
                summarydata.append(sent)

    return ' '.join(docdata), ' '.join(summarydata)

class XSumWriter:
    def __init__(self, args):
        self.data_dir = os.path.join(args.input_dir, 'bbc-summary-data/')
        self.splits_file = os.path.join(args.input_dir, 'XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json')
        self.output_dir = args.output_dir

    def write(self):
        split_dict = json.loads(open(self.splits_file).read())
        split_types = ["test", "validation", "train"]

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        count = 0

        for split in split_types:
            docpath = os.path.join(self.output_dir, '{}.src'.format(split))
            summarypath = os.path.join(self.output_dir, '{}.tgt'.format(split))

            with open(docpath, mode='w+') as df, open(summarypath, mode='w+') as sf:
                for docid in split_dict[split]:
                    filepath = os.path.join(self.data_dir, docid + '.summary')
                    with open(filepath, mode='r') as f:
                        text = f.read().splitlines()
                        doc, summary = process_line(text)
                        df.write(doc + '\n')
                        sf.write(summary + '\n')
                    if count % 1000 == 0:
                        print(count)
                    count += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default='datasets/xsum/unprocessed/',
    )
    parser.add_argument(
        "--output-dir",
        default='datasets/xsum/preprocessed/',
    )
    args = parser.parse_args()

    writer = XSumWriter(args)
    writer.write()

if __name__ == "__main__":
    main()