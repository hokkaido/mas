import argparse
import transformers
from classification_model import ClassificationModel

import pandas as pd
from sklearn.utils import class_weight
from ray import tune
import os

labels = {
    '__label__0': 0,
    '__label__1': 1,
}

def load_df(fp, frac=1):
    rows = []
    with open(fp, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            example = line.split(" ", 1)
            if len(example) == 2:
                rows.append({'text': example[1], 'label': labels[example[0]]})
    df = pd.DataFrame(data=rows, columns=['text', 'label'])
    return df.sample(frac=frac).reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        default='datasets/constrained_classification/k16-window/k16.train',
    )
    parser.add_argument(
        "--eval-path",
        default='datasets/constrained_classification/k16-window/k16.valid',
    )

    args = parser.parse_args()

    train_df = load_df(args.train_path, 0.005)
    eval_df = load_df(args.eval_path, 0.005)

    print()

    model_args = {
        'evaluate_during_training': True,
        'log_tune': True,
        'train_batch_size': 16,
        'gradient_accumulation_steps': 8,
        'eval_batch_size': 16,
        'num_train_epochs': 10,
        'eval_steps': 250,
        'save_steps': 10000000,
        'cache_dir': os.path.join(os.getcwd(), 'cache_dir')
    }

    space = {
        'warmup_ratio': tune.grid_search([0.02, 0.04]),
        'learning_rate': tune.grid_search([1e-4, 1e-5, 5e-5]),
        'max_seq_length': tune.grid_search([128, 256, 512])
    }

    resources_per_trial = {
        "cpu": 8,
        "gpu": 1,
    }

    target_count = train_df.label.value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])

    class_weights = class_weight.compute_class_weight('balanced',
                                                 [0, 1],
                                                 train_df.label)

    def train_model(config):

        model = ClassificationModel('bert', 'bert-base-uncased', num_labels=2, weight=class_weights)

        args = model_args.copy()
        args['warmup_ratio'] = config['warmup_ratio']
        args['learning_rate'] = config['learning_rate']
        args['max_seq_length'] = config['max_seq_length']

        model.train_model(train_df, eval_df=eval_df, args=args)           

    print(class_weights)

    analysis = tune.run(
        train_model, config=space, resources_per_trial=resources_per_trial)

    print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))

if __name__ == "__main__":
    main()