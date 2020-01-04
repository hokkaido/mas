import re
import csv
import collections
import itertools
import random

Example = collections.namedtuple('Example', ['source', 'target'])
Summary = collections.namedtuple('Summary', ['model_name', 'source', 'summary'])
BestWorstPairing = collections.namedtuple('BestWorstPairing', ['source', 'model_name_a', 'summary_a', 'model_name_b', 'summary_b'])

"""
Preparing documents... 0 line(s) ignored                                                                                                                              
Running ROUGE...
---------------------------------------------
1 ROUGE-1 Average_R: 0.44844 (95%-conf.int. 0.44592 - 0.45093)
1 ROUGE-1 Average_P: 0.40209 (95%-conf.int. 0.39940 - 0.40490)
1 ROUGE-1 Average_F: 0.41271 (95%-conf.int. 0.41032 - 0.41493)
---------------------------------------------
1 ROUGE-2 Average_R: 0.20348 (95%-conf.int. 0.20094 - 0.20622)
1 ROUGE-2 Average_P: 0.18320 (95%-conf.int. 0.18065 - 0.18584)
1 ROUGE-2 Average_F: 0.18751 (95%-conf.int. 0.18517 - 0.19014)
---------------------------------------------
1 ROUGE-L Average_R: 0.41458 (95%-conf.int. 0.41212 - 0.41728)
1 ROUGE-L Average_P: 0.37218 (95%-conf.int. 0.36945 - 0.37496)
1 ROUGE-L Average_F: 0.38180 (95%-conf.int. 0.37950 - 0.38410)

Elapsed time: 133.541 seconds
"""

metrics = ['ROUGE-1 Average_F', 'ROUGE-2 Average_F', 'ROUGE-L Average_F']

def parse_rouge_result(lines):
    result = {}

    for line in lines:
        for metric in metrics:
            if metric in line:
                tokens = line.replace(')', ' ').split()
                result[metric] = {
                    'avg': float(tokens[3]),
                    'lower': float(tokens[5]),
                    'upper': float(tokens[7])
                }
                #print(result[metric]['avg'] - result[metric]['lower'])
                #print(result[metric]['upper'] - result[metric]['avg'])
    return result

def read_rouge(fp='rouge.txt'):
    with open(fp) as f:
        return parse_rouge_result(f.readlines())

class SummaryHolder:
    def __init__(self, gold_examples):
        if gold_examples is None:
            raise Exception('Please provide a list of gold examples')

        self.model_summaries = {}

        for example in gold_examples:
            self.model_summaries[example.source] = [Summary(model_name='gold', source=example.source, summary=example.target)]

    def add_model(self, model_name, examples):
        if model_name == 'gold':
            raise Exception('gold is a reserved model name')

        for example in examples:
            if example.source not in self.model_summaries:
                raise Exception('Example source unknown: {}'.format(example.source))
            
            self.model_summaries[example.source].append(Summary(model_name=model_name, source=example.source, summary=example.target))


    def create_pairings(self):
        def collate(source):
            return itertools.combinations(self.model_summaries[source], 2)

        result = []

        for source in self.model_summaries:
            pairings = collate(source)
            for pairing in pairings:
                a, b = pairing
                assert a.source == b.source
                if a.model_name != b.model_name:
                    if random.random() < 0.5: # randomize order
                        a, b = b, a
                    result.append(BestWorstPairing(a.source, a.model_name, a.summary, b.model_name, b.summary))

        return result

def create_fake_examples(model_name):
    examples = []
    for c in 'abcdefghij':
        examples.append(Example(source='{}-source'.format(c), target='{}-{}-target'.format(model_name, c)))
    return examples

def write_pairings(pairings, out='pairings.csv'):
    with open(out, 'w') as f:
        w = csv.writer(f)
        w.writerow(('source', 'model_name_a', 'summary_a', 'model_name_b', 'summary_b'))
        w.writerows([(p.source, p.model_name_a, p.summary_a, p.model_name_b, p.summary_b) for p in pairings])


def test():
    gold = create_fake_examples('gold')
    model_1 = create_fake_examples('model-1')
    model_2 = create_fake_examples('model-2')
    model_3 = create_fake_examples('model-3')
    model_4 = create_fake_examples('model-4')
    model_5 = create_fake_examples('model-5')

    holder = SummaryHolder(gold)

    holder.add_model('model-1', model_1)
    holder.add_model('model-2', model_2)
    holder.add_model('model-3', model_3)

    pairings = holder.create_pairings()
    print(len(pairings))
    random.shuffle(pairings)
    #write_pairings(pairings)

def latex_eval():
    import os
    for root, dirs, files in os.walk("eval/exp2"):
        for file in files:
            if file.endswith("rouge.txt"):
                rp = os.path.join(root, file)
                metrics = read_rouge(rp)

                print('{} & {} & ({} - {}) & {} & ({} - {}) & {} & ({} - {})'.format(
                    root, 
                    round(100 * metrics['ROUGE-1 Average_F']['avg'], 2),
                    round(100 * metrics['ROUGE-1 Average_F']['lower'], 2),
                    round(100 * metrics['ROUGE-1 Average_F']['upper'], 2),
                    round(100 * metrics['ROUGE-2 Average_F']['avg'], 2),
                    round(100 * metrics['ROUGE-2 Average_F']['lower'], 2),
                    round(100 * metrics['ROUGE-2 Average_F']['upper'], 2),
                    round(100 * metrics['ROUGE-L Average_F']['avg'], 2),
                    round(100 * metrics['ROUGE-L Average_F']['lower'], 2),
                    round(100 * metrics['ROUGE-L Average_F']['upper'], 2)
                ))
#latex_eval()


test()