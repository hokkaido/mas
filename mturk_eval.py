import random
from mas import print_mturk_eval_results

def main(seed=9372):
    random.seed(seed)
    print_mturk_eval_results(file='eval/exp2-xsum/Batch_3894913_batch_results.csv')


if __name__ == "__main__":
    main()