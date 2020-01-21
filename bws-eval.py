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

import spacy

nlp = spacy.load('en_core_web_lg')
import truecase

COMPANIES = ['CNN', 'EU', 'AI', 'BT', 'NHS', 'BBC', 'PKK', 'CES', 'UK', 'FA', 'IFA', 'FSB']

SPECIAL = {
    'ais': 'AIs'
}

class SummaryPicker:
    def  __init__(self, exp_path):
        
        self.exp_path = exp_path
        self.load_experiment()
        self.detokenizer = MosesDetokenizer(lang='en')
        self.truecaser = MosesTruecaser(load_from='sm.cnndm.tc.model')

    def cleanup(self, line, append=False):

        # todo, use proper regex
        line = line.replace('- lrb -', '(')
        line = line.replace('- rrb -', ')')
        line = line.replace('- lsb -', '[')
        line = line.replace('- rsb -', ']')
        line = line.replace('`', "'")
        line = self.detokenizer.detokenize(line.split(' '))

        # line = line.replace(" 's ", "'s ")
        # line = line.replace(" 'd ", "'d ")
        # line = line.replace("' s ", "'s ")
        # line = line.replace(" n '", "n'")
        # line = line.replace(" n' ", "n'")
        line = line.replace(" - - ", " -- ")
        line = line.replace(" - ", "-")
        line = re.sub(r', (\d{3})', r',\1', line)
        line = line.replace("i' m ", "i'm ")
        line = line.replace(" 'll ", "'ll ")
        line = line.replace("' ll ", "'ll ")
        line = re.sub(r" '([a-zA-Z]{1}) ", r"'\1 ", line)
        line = re.sub(r"' ([a-zA-Z]{1}) ", r"'\1 ", line)
        line = re.sub(r" ([a-zA-Z]{1})' ", r" \1'", line)
        line = re.sub(r"(you|they)(' re )", r"\1're ", line)
        line = re.sub(r"(\$\d+\.) (\d+)", r"\1\2", line)
        line = re.sub(r"(\d{1,2}): (\d{1,2}) (am|pm)", r"\1:\2\3", line)
        line = line.replace(" n't ", "n't ")
        line = line.replace(" 've ", "'ve ")

        doc = nlp(line)

        lines = []

        def repr_word(tok):
            txt = tok.text_with_ws
            if tok.text in SPECIAL:
                txt = txt.replace(tok.text, SPECIAL[tok.text])
            elif tok.is_sent_start or tok.ent_type_ in ['PERSON', 'ORG', 'PRODUCT', 'GPE', 'LOC', 'FAC', 'NORP', 'EVENT', 'WORK_OF_ART']:
                txt = txt.capitalize()
            if tok.text.upper() in COMPANIES:
                txt = txt.upper()
            return txt

        for tok in doc:
            lines.append(repr_word(tok))

        line = ''.join(lines)
        if not line.endswith('.') and append:
            line += ' ...'

        return line
        # line = line.replace(" 'm ", "'m ")
        # line = line.replace("' m ", "'m ")

    def cleanup_samples(self, samples):
        return [self.cleanup_sample(sample) for sample in samples]

    def cleanup_sample(self, sample):
        sample['source'] = self.cleanup(sample['source'], append=True)
        sample['summaries'] = { model: self.cleanup(line) for model, line in sample['summaries'].items()}
        return sample
       
    def load_corpus(self, text_path):
        with open(text_path,'r') as f:
            return f.read().splitlines()
        
    def sample(self, n=15, clean=True):
        indices = random.sample(range(len(self.source)), n)
        samples = []
        for idx in indices:
            sample = {}
            sample['index'] = idx
            sample['source'] = self.source[idx]
            sample['summaries'] = {}
            sample['summaries']['gold'] = self.gold[idx]
            for key, docs in self.results.items():
                sample['summaries'][key] = docs[idx]
            
            if clean:
                sample = self.cleanup_sample(sample)
            samples.append(sample)
            
        
        return samples
        
    def load_experiment(self):
        self.results = {}
        
        logging.info('Loading source articles')
        
        self.source = self.load_corpus(os.path.join(self.exp_path, 'src.txt'))
        self.gold = None
        
        for d in os.listdir(self.exp_path):
            res_path = os.path.join(self.exp_path, d)
            if os.path.isdir(res_path):
                if self.gold is None:
                    logging.info('Loading gold summaries')
                    self.gold = self.load_corpus(os.path.join(res_path, 'tar.txt'))
                    
                    assert len(self.gold) == len(self.source)

                logging.info('Loading {}'.format(res_path))
                corpus = self.load_corpus(os.path.join(res_path, 'hyp.txt'))
                assert len(corpus) == len(self.source)
                self.results[os.path.basename(res_path)] = corpus
                

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
            return itertools.combinations(self.model_summaries[source], 3)

        result = []

        for source in self.model_summaries:
            pairings = collate(source)
            for pairing in pairings:
                a, b, c = pairing
                assert a.source == b.source == c.source

                l = list(pairing)
                random.shuffle(l)
                a, b, c = l
                result.append(BestWorstPairing(a.source, a.model_name, a.summary, b.model_name, b.summary, c.model_name, c.summary))

        return result

def create_fake_examples(model_name):
    examples = []
    for c in 'abcdefghij':
        examples.append(Example(source='{}-source'.format(c), target='{}-{}-target'.format(model_name, c)))
    return examples

def write_pairings(pairings, out='pairings.csv'):
    with open(out, 'w') as f:
        w = csv.writer(f)
        w.writerow(('source', 'model_name_a', 'summary_a', 'model_name_b', 'summary_b', 'model_name_c', 'summary_c'))
        w.writerows([(p.source, p.model_name_a, p.summary_a, p.model_name_b, p.summary_b, p.model_name_c, p.summary_c) for p in pairings])


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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main(exp_path='eval/exp1', num_evaluators=3, models_to_eval=['cnndm-reference', 'cnndm-entities-encoder', 'cnndm-entities-encoder-segments-encoder'], seed=9372):
    random.seed(seed)
    picker = SummaryPicker(exp_path)

    samples = picker.sample(16, clean=False)

    with open(os.path.join(exp_path, 'human_eval_raw_samples.json'), 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)

    samples = picker.cleanup_samples(samples)

    with open(os.path.join(exp_path, 'human_eval_clean_samples.json'), 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)
    examples_by_model = collections.defaultdict(list)

    for sample in samples:
        for model, summary in sample['summaries'].items():
            example = Example(sample['source'], summary)
            examples_by_model[model].append(example)

    holder = SummaryHolder(examples_by_model['gold'])
    for model, examples in examples_by_model.items():
        if model == 'gold' or model not in models_to_eval:
            continue
        holder.add_model(model, examples)


    
    for k in range(num_evaluators):
        pairings = holder.create_pairings()
        random.shuffle(pairings)
        write_pairings(pairings, os.path.join(exp_path, 'human_eval_{}.csv'.format(k)))

if __name__ == "__main__":
    exp_path='eval/exp2-xsum'
    num_evaluators=3
    models_to_eval=['xsum-vanilla', 'xsum-entities-encoder', 'xsum-entities-encoder-segments-encoder']
    main(exp_path=exp_path, num_evaluators=num_evaluators, models_to_eval=models_to_eval, seed=2020)