import logging
logging.basicConfig(level=logging.WARNING)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from hpbandster.core.worker import Worker

from typing import Mapping, Iterable
from itertools import product
import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
import numpy as np
import copy
import re
import files2rouge as f2r
import tempfile
import os
import pyrouge
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

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


class TrueRougeScorer:
    def score(self, pairs):
        with tempfile.NamedTemporaryFile(mode = "a+") as h, tempfile.NamedTemporaryFile(mode = "a+") as t:
            for pair in pairs:
                target, hypo = pair
                print(' '.join(target), file=t, flush=True)
                print(' '.join(hypo), file=h, flush=True)
            
            output = run_p2r(t.name, h.name)

        return parse_rouge(output)

def setup_model(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

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

    return args, models, task

    
class BeamWorker(Worker):
    def __init__(self, args, scorer, **kwargs):
        super().__init__(**kwargs)
        args, models, task = setup_model(args)
        self.task = task
        self.args = args
        self.models = models
        self.scorer = scorer
        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(self.args.replace_unk)
        self.samples = []

        itr = self.get_batch_iterator().next_epoch_itr(shuffle=False)

        for _ in range(int(args.hpb_max_budget)):
            self.samples.append(next(itr))

    def make_fit_args(self, config):
        fit_args = copy.deepcopy(self.args)
        for k, v in config.items():
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

    def compute(self, config, budget, *args, **kwargs):

        summaries = []      

        for i in range(int(budget)):
            sample = self.samples[i]
            summaries.extend(self.gen_candidates(config, sample))
        score = self.scorer.score(summaries)

        print(score)

        return {
            'loss': -score['ROUGE-1-F (avg)'],
            'info': score
        }

    def gen_candidates(self, config, sample):
        use_cuda = torch.cuda.is_available() and not self.args.cpu

        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if 'net_input' not in sample:
            return

        src_dict = self.task.source_dictionary
        tgt_dict = self.task.target_dictionary

        regex = r" ##"
        args = self.make_fit_args(config)
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

def get_cnndm_space():
    cs = CS.ConfigurationSpace(seed=1)
    beam = CSH.CategoricalHyperparameter('beam', choices=[4, 5])
    ngram = CSH.CategoricalHyperparameter('no_repeat_ngram_size', choices=[3, 4])
    lenpen = CSH.UniformFloatHyperparameter('lenpen', lower=0.5, upper=2.0, q=0.1)   
    min_len = CSH.UniformIntegerHyperparameter('min_len', lower=40, upper=60)
    max_len_b = CSH.UniformIntegerHyperparameter('max_len_b', lower=100, upper=200, q=5)

    cs.add_hyperparameters([beam, ngram, lenpen, min_len, max_len_b])
    return cs

def get_xsum_space():
    cs = CS.ConfigurationSpace(seed=1)
    beam = CSH.CategoricalHyperparameter('beam', choices=[4, 5])
    ngram = CSH.CategoricalHyperparameter('no_repeat_ngram_size', choices=[3, 4])
    lenpen = CSH.UniformFloatHyperparameter('lenpen', lower=0.5, upper=2.0, q=0.1)   
    min_len = CSH.UniformIntegerHyperparameter('min_len', lower=5, upper=40)
    max_len_b = CSH.UniformIntegerHyperparameter('max_len_b', lower=0, upper=50, q=5)

    cs.add_hyperparameters([beam, ngram, lenpen, min_len, max_len_b])
    return cs

def infer_run_id_from_args(args):
    return os.path.basename(os.path.dirname(args.path))

def main(args):

    scorer = TrueRougeScorer()

    run_id = infer_run_id_from_args(args)

    run_dir = os.path.join(args.hpb_runs_dir, run_id)

    if args.hpb_worker:
        w = BeamWorker(args, scorer, nameserver='127.0.0.1', run_id=run_id)
        w.run(background=False)
        exit(0)

    result_logger = hpres.json_result_logger(directory=run_dir, overwrite=False)

    # Step 1: Start a nameserver
    # Every run needs a nameserver. It could be a 'static' server with a
    # permanent address, but here it will be started for the local machine with the default port.
    # The nameserver manages the concurrent running workers across all possible threads or clusternodes.
    # Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
    NS = hpns.NameServer(run_id=run_id, host='127.0.0.1', port=None)
    NS.start()

    # Step 2: Start a worker
    # Now we can instantiate a worker, providing the mandatory information
    # Besides the sleep_interval, we need to define the nameserver information and
    # the same run_id as above. After that, we can start the worker in the background,
    # where it will wait for incoming configurations to evaluate.



    # Step 3: Run an optimizer
    # Now we can create an optimizer object and start the run.
    # Here, we run BOHB, but that is not essential.
    # The run method will return the `Result` that contains all runs performed.
    bohb = BOHB(configspace = get_cnndm_space(),
                run_id = run_id, nameserver='127.0.0.1',
                result_logger=result_logger,
                min_budget=args.hpb_min_budget, max_budget=args.hpb_max_budget
            )
    res = bohb.run(n_iterations=args.hpb_n_iterations, min_n_workers=args.hpb_n_workers)

    # Step 4: Shutdown
    # After the optimizer run, we must shutdown the master and the nameserver.
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # Step 5: Analysis
    # Each optimizer returns a hpbandster.core.result.Result object.
    # It holds informations about the optimization run like the incumbent (=best) configuration.
    # For further details about the Result object, see its documentation.
    # Here we simply print out the best config and some statistics about the performed runs.
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.'%(sum([r.budget for r in res.get_all_runs()])/args.hpb_max_budget))


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument('--hpb_min_budget',   type=float, help='Minimum budget used during the optimization.',    default=2)
    parser.add_argument('--hpb_max_budget',   type=float, help='Maximum budget used during the optimization.',    default=16)
    parser.add_argument('--hpb_n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=16)
    parser.add_argument('--hpb_n_workers', type=int,   help='Number of workers to run in parallel.', default=2)
    parser.add_argument('--hpb_worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--hpb_runs_dir', type=str,   help='location of runs dir', default='runs')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    torch.set_printoptions(profile="short")
    torch.set_printoptions(threshold=50)
    cli_main()