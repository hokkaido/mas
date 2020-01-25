#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""
from typing import Mapping, Iterable
from itertools import product
import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
import numpy as np
import copy
from rouge import rouge_n_sentence_level, rouge_l_summary_level
import re
import files2rouge as f2r
import tempfile
import os
import pyrouge
import logging

parameters = {'beam': [4], 'no_repeat_ngram_size': [3], 'lenpen': [0.5, 1.0, 2], 'min_len': [45, 50, 55], 'max_len_b': [125, 150, 200]}

def run_p2r(summ_path,
        ref_path,
        rouge_args=None,
        verbose=False,
        saveto=None,
        eos=".",
        ignore_empty=False,
        stemming=False):
    s = f2r.settings.Settings()
    s._load()
    dirpath = tempfile.mkdtemp()
    sys_root, model_root = [os.path.join(dirpath, _)
                            for _ in ["system", "model"]]

    print("Preparing documents...", end=" ")
    f2r.utils.mkdirs([sys_root, model_root])
    ignored = f2r.utils.split_files(model_file=ref_path,
                                system_file=summ_path,
                                model_dir=model_root,
                                system_dir=sys_root,
                                eos=eos,
                                ignore_empty=ignore_empty)
    log_level = logging.ERROR if not verbose else None
    r = pyrouge.Rouge155(rouge_dir=os.path.dirname(s.data['ROUGE_path']), log_level=log_level)
    r.system_dir = sys_root
    r.model_dir = model_root
    r.system_filename_pattern = r's.(\d+).txt'
    r.model_filename_pattern = 'm.[A-Z].#ID#.txt'
    data_arg = "-e %s" % s.data['ROUGE_data']

    if not rouge_args:
        rouge_args = [
            '-c', 95,
            '-r', 1000,
            '-n', 2,
            '-a']
        if stemming:
            rouge_args.append("-m")

        rouge_args_str = " ".join([str(_) for _ in rouge_args])
    else:
        rouge_args_str = rouge_args
    rouge_args_str = "%s %s" % (data_arg, rouge_args_str)
    output = r.convert_and_evaluate(rouge_args=rouge_args_str)

    return output

def parse_rouge(text):

    r1 = float(text.partition('ROUGE-1 Average_F: ')[2][:7])
    r2 = float(text.partition('ROUGE-2 Average_F: ')[2][:7])
    r3 = float(text.partition('ROUGE-L Average_F: ')[2][:7])
    return {
        'ROUGE-1-F (avg)': r1,
        'ROUGE-2-F (avg)': r2,
        'ROUGE-L-F (avg)': r3,
        }

0.44341
class TrueRougeScorer:
    def score(self, pairs):
        with tempfile.NamedTemporaryFile(mode = "a+") as h, tempfile.NamedTemporaryFile(mode = "a+") as t:
            for pair in pairs:
                target, hypo = pair
                print(' '.join(target), file=t, flush=True)
                print(' '.join(hypo), file=h, flush=True)
            
            output = run_p2r(t.name, h.name)

        return parse_rouge(output)

class RougeScorer:
    def score(self, pairs):
        rouges_1 = []
        rouges_2 = []
        rouges_l = []
        for pair in pairs:
            target, hypo = pair
            # Calculate ROUGE-2.
            _, _, rouge_1 = rouge_n_sentence_level(hypo, target, 1)
            _, _, rouge_2 = rouge_n_sentence_level(hypo, target, 2)
            _, _, rouge_l = rouge_l_summary_level(hypo, target)
            rouges_1.append(rouge_1)
            rouges_2.append(rouge_2)
            rouges_l.append(rouge_l)
        return {
            'ROUGE-1-F (avg)': np.average(rouges_1),
            'ROUGE-2-F (avg)': np.average(rouges_2),
            'ROUGE-L-F (avg)': np.average(rouges_l),
        }

class ParameterGrid:
    def __init__(self, param_grid):
        if not isinstance(param_grid, (Mapping, Iterable)):
            raise TypeError('Parameter grid is not a dict or '
                            'a list ({!r})'.format(param_grid))

        if isinstance(param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            param_grid = [param_grid]

        # check if all entries are dictionaries of lists
        for grid in param_grid:
            if not isinstance(grid, dict):
                raise TypeError('Parameter grid is not a '
                                'dict ({!r})'.format(grid))
            for key in grid:
                if not isinstance(grid[key], Iterable):
                    raise TypeError('Parameter grid value is not iterable '
                                    '(key={!r}, value={!r})'
                                    .format(key, grid[key]))

        self.param_grid = param_grid

    def __iter__(self):
        """Iterate over the points in the grid.
        Returns
        -------
        params : iterator over dict of string to any
            Yields dictionaries mapping each estimator parameter to one of its
            allowed values.
        """
        for p in self.param_grid:
            # Always sort the keys of a dictionary, for reproducibility
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

class GridSearch:
    def __init__(self, params, args, task, models, scorer):
        self.param_grid = ParameterGrid(params)
        self.task = task
        self.args = args
        self.models = models
        self.scorer = scorer
        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(self.args.replace_unk)


    def make_fit_args(self, params):
        fit_args = copy.deepcopy(self.args)
        for k, v in params.items():
            vars(fit_args)[k] = v
        return fit_args

    def get_batch_iterator(self):
        return self.task.get_batch_iterator(
                dataset=self.task.dataset(self.args.gen_subset),
                max_tokens=self.args.max_tokens,
                max_sentences=self.args.max_sentences,
                max_positions=utils.resolve_max_positions(
                    self.task.max_positions(),
                    *[model.max_positions() for model in self.models]
                ),
                ignore_invalid_inputs=self.args.skip_invalid_size_inputs_valid_test,
                required_batch_size_multiple=self.args.required_batch_size_multiple,
                num_shards=self.args.num_shards,
                shard_id=self.args.shard_id,
                num_workers=self.args.num_workers,
            )

    def fit(self, n = 8):

        scores = []

        samples = []
        itr = self.get_batch_iterator().next_epoch_itr(shuffle=False)
        for _ in range(n):
            samples.append(next(itr))
        
        for params in self.param_grid:
            print(params)
            
            print('init')
            to_score = []

            for sample in samples:
                to_score.extend(self.gen_candidates(params, sample))

            score = self.scorer.score(to_score)
            scores.append((params, score))
            print(score)
        return scores

    def gen_candidates(self, params, sample):

        use_cuda = torch.cuda.is_available() and not self.args.cpu
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            return

        src_dict = self.task.source_dictionary
        tgt_dict = self.task.target_dictionary

        regex = r" ##"
        args = self.make_fit_args(params)
        generator = self.task.build_generator(args)

        prefix_tokens = None
        if args.prefix_size > 0:
            prefix_tokens = sample['target'][:, :args.prefix_size]

        hypos = self.task.inference_step(generator, self.models, sample, prefix_tokens)
        num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
        
        to_score = []

        for i, sample_id in enumerate(sample['id'].tolist()):
            # Remove padding
            src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
            target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()
            src_str = src_dict.string(src_tokens, args.remove_bpe)
            target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][:args.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=self.align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )

                # Score only the top hypothesis
                if j == 0:
                    #if align_dict is not None or args.remove_bpe is not None:
                    #    # Convert back to tokens for evaluation with unk replacement and/or without BPE
                    #    target_tokens = tgt_dict.encode_line(target_str, add_if_not_exist=True)
                    #if hasattr(scorer, 'add_string'):
                    #    self.scorer.add_string(target_str, hypo_str)
                    #else:
                    #self.scorer.add(target_tokens, hypo_tokens)
                    # print(hypo_str)
                    # print(target_str)
                    hypo_str = re.sub(regex, "", hypo_str, 0, re.MULTILINE)
                    target_str = re.sub(regex, "", target_str, 0, re.MULTILINE)
                    to_score.append((target_str.split(), hypo_str.split()))

        return to_score

def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            need_attn=False,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator

    scorer = TrueRougeScorer()
    gridsearch = GridSearch(parameters, args, task, models, scorer)

    scores = gridsearch.fit()

    print('top scores')

    asc_scores = sorted(scores, key=lambda x: (x[1]['ROUGE-L-F (avg)'], x[1]['ROUGE-2-F (avg)'], x[1]['ROUGE-1-F (avg)']))
    for score in asc_scores:
        print(score[0])
        print(score[1])
        print('------')


def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    torch.set_printoptions(profile="short")
    torch.set_printoptions(threshold=50)
    cli_main()
