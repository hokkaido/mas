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

metrics = ['ROUGE-1 Average_F', 'ROUGE-2 Average_F', 'ROUGE-L Average_F', 'ROUGE-1 Average_R', 'ROUGE-2 Average_R', 'ROUGE-L Average_R']

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

def latex_eval(fp):
    import os
    for root, dirs, files in os.walk(fp):
        dirs.sort()
        for file in sorted(files):
            if file.endswith("rouge.txt"):
                rp = os.path.join(root, file)
                metrics = read_rouge(rp)

                print('{} & {} & {} - {} & {} & {} - {} & {} & {} - {}'.format(
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
