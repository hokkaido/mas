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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default='datasets/xsum/unprocessed/bbc-summary-data/',
    )
    parser.add_argument(
        "--splits",
        default='datasets/xsum/unprocessed/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json',
    )
    parser.add_argument(
        "--output",
        default='datasets/xsum/preprocessed/',
    )
    args = parser.parse_args()

    split_dict = json.loads(open(args.splits).read())
    split_types = ["test", "validation", "train"]


    count = 0

    for split in split_types:
        docpath = os.path.join(args.output, '{}.src'.format(split))
        summarypath = os.path.join(args.output, '{}.tgt'.format(split))

        with open(docpath, mode='w+') as df, open(summarypath, mode='w+') as sf:

            for docid in split_dict[split]:
                filepath = os.path.join(args.input, docid + '.summary')
                with open(filepath, mode='r') as f:
                    text = f.read().splitlines()
                    doc, summary = process_line(text)
                    df.write(doc + '\n')
                    sf.write(summary + '\n')
                if count % 1000 == 0:
                    print(count)
                count += 1

if __name__ == "__main__":
    main()