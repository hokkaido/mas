import re
import csv
import collections
import itertools
import random
import logging
import os
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sacremoses import MosesDetokenizer, MosesTruecaser
import json
Example = collections.namedtuple('Example', ['source', 'target'])
Summary = collections.namedtuple('Summary', ['model_name', 'source', 'summary'])
BestWorstPairing = collections.namedtuple('BestWorstPairing', ['source', 'model_name_a', 'summary_a', 'model_name_b', 'summary_b', 'model_name_c', 'summary_c'])
import numpy as np
import truecase
import matplotlib.pyplot as plt
import pandas as pd

def read_results(file='eval/exp1/Batch_3883616_batch_results.csv'):

        df = pd.read_csv(file)
        df = df.fillna(False)

        print(df.quantile(0.1))
        models = {}

        print(df.describe()['WorkTimeInSeconds'])

        df = df[df['RequesterFeedback'] == False]

        df.boxplot('WorkTimeInSeconds')

        print(df.describe()['WorkTimeInSeconds'])

        df.boxplot('WorkTimeInSeconds', by='WorkerId')
        

        suspect_workers = df[df['WorkTimeInSeconds'] < 60]['WorkerId'].unique()

    
        print(suspect_workers)

        for col in df.columns:
            if col.startswith('Answer.best.'):
                model_name = col.partition('Answer.best.')[2]
                worst_col = 'Answer.worst.{}'.format(model_name)
                models[model_name] = { 'best': 0, 'worst': 0 }
                models[model_name]['best'] = df[col].values.sum()
                models[model_name]['worst'] = df[worst_col].values.sum()

        print(models)
        for model in models:
            print(model)
            ratio = len(df) / len(models)
            print(models[model]['best']/ratio - models[model]['worst']/ratio)

        suspect_mask = df['WorkerId'].isin(suspect_workers)
        suspect_work = df[suspect_mask]
        analyze_rows(suspect_work)

        plt.show()

def without_outliers(df):
    Q1 = df['WorkTimeInSeconds'].quantile(0.25)
    Q3 = df['WorkTimeInSeconds'].quantile(0.75)
    IQR = Q3 - Q1

    filter = (df['WorkTimeInSeconds'] >= Q1 - 0.75 * IQR) & (df['WorkTimeInSeconds'] <= Q3 + 0.75 *IQR)
    print(df.loc[filter].describe()['WorkTimeInSeconds'])

def stats(file='eval/exp1/Batch_3883616_batch_results.csv'):

        df = pd.read_csv(file)
        df = df.fillna(False)

        models = {}

        df.boxplot('WorkTimeInSeconds', by='WorkerId')
        df = df[df['RequesterFeedback'] == False]

        print('Containing {} evaluations', len(df))

        for col in df.columns:
            if col.startswith('Answer.best.'):
                model_name = col.partition('Answer.best.')[2]
                worst_col = 'Answer.worst.{}'.format(model_name)
                models[model_name] = { 'best': 0, 'worst': 0 }
                models[model_name]['best'] = df[col].values.sum()
                models[model_name]['worst'] = df[worst_col].values.sum()

        print(models)
        for model in models:
            print(model)
            ratio = len(df) / len(models)
            print(models[model]['best']/ratio - models[model]['worst']/ratio)


def analyze_rows(df):

    dfg = df.groupby('WorkerId')

    print(dfg.describe()['WorkTimeInSeconds'])



    counts = [len(v) for k, v in dfg]
    total = float(sum(counts))
    cases = len(counts)

    widths = [c/total for c in counts]  

    cax = df.boxplot(column='WorkTimeInSeconds', by='WorkerId', widths=widths)
    cax.set_xticklabels(['%s\n$n$=p2%d'%(k, len(v)) for k, v in dfg])



def main(seed=9372):
    random.seed(seed)
    stats(file='eval/exp2-xsum/Batch_3894913_batch_results.csv')


if __name__ == "__main__":
    main()