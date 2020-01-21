#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter



import pandas as pd


@torch.no_grad()
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
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()

    num_sentences = 0
    has_target = True

    all_probs = []
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for step, sample in enumerate(t):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue
            encoder_inputs = {
                k: v for k, v in sample['net_input'].items()
                if k != 'prev_output_tokens'
            }

            targets = sample['target']

            for model in models:

                for i, sample_id in enumerate(sample['id'].tolist()):

                    encoder_input = {
                        k: v[i:i+1,:] for k, v in encoder_inputs.items()
                        if v is not None and k != 'src_lengths'
                    }

                    encoder_input['src_lengths'] = i + 1

                    encoder_out = model.encoder.forward(**encoder_input)

                    target_tokens = utils.strip_pad(targets[i, :], tgt_dict.pad()).cpu()

                    decoder_targets = torch.unsqueeze(target_tokens, 0)
                    decoder_out = model.decoder.forward(decoder_targets, encoder_out)
                    probs = model.get_normalized_probs(decoder_out, log_probs=False)

                    sample_probs = []
                    for j in range(len(target_tokens)):

                        sample_probs.append(probs[:,j,:].flatten()[target_tokens[j]].item())
                    all_probs.append(sample_probs)
                    # print(target_tokens.shape)
                    # for j in range(len(target_tokens)):
                        
                    #     decoder_targets = torch.unsqueeze(target_tokens[:j+1], 0)
                    #     decoder_out = model.decoder.forward(decoder_targets, encoder_out)
                    #     probs = model.get_normalized_probs(decoder_out, log_probs=False)
                    #     print('---------------')

                    #     print(probs[:,0,:].flatten()[target_tokens[0]])
                    #     print(probs[:,j,:].flatten()[target_tokens[j]])
                    #     print('---------------')

            if step == 16:
                df = pd.DataFrame.from_records(all_probs)
                df.to_csv('probs.csv', index=False)
       

def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    torch.set_printoptions(profile="short")
    torch.set_printoptions(threshold=50)
    cli_main()