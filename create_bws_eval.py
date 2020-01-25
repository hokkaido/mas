import argparse
from mas import create_bws_eval

def main(args):
    create_bws_eval(
        exp_path=args.experiment_dir, 
        num_evaluators=args.num_evaluators, 
        num_samples = args.samples, 
        models_to_eval=args.models, 
        seed=args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_evaluators', type=int, default=3)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--models', type=str, nargs='+', required=True)
    parser.add_argument('--experiment-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=2020)
    args = parser.parse_args()
    main(args)

