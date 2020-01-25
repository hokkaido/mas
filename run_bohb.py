
import torch
from fairseq import options
from mas import run_bohb

import logging
logging.basicConfig(level=logging.WARNING)

def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument('--hpb_min_budget',   type=float, help='Minimum budget used during the optimization.',    default=1)
    parser.add_argument('--hpb_max_budget',   type=float, help='Maximum budget used during the optimization.',    default=32)
    parser.add_argument('--hpb_n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=32)
    parser.add_argument('--hpb_n_workers', type=int,   help='Number of workers to run in parallel.', default=2)
    parser.add_argument('--hpb_worker', help='Flag to turn this into a worker process', action='store_true')
    parser.add_argument('--hpb_overwrite_run', help='Flag to overwrite run', action='store_true')
    parser.add_argument('--hpb_runs_dir', type=str,   help='location of runs dir', default='runs')
    parser.add_argument('--hpb_run_id', type=str,   help='run id')
    parser.add_argument('--hpb_metric', type=str,   help='Which metric to optimize', default='ROUGE-1-F (avg)')
    parser.add_argument('--hpb_config', type=str,   help='config space, either cnndm, xsum or duc2004')
    args = options.parse_args_and_arch(parser)
    run_bohb(args)


if __name__ == '__main__':
    torch.set_printoptions(profile="short")
    torch.set_printoptions(threshold=50)
    cli_main()