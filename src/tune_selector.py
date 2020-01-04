import argparse
import transformers
from classification_model import ClassificationModel
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import ray
from ray import tune
from ray.tune.util import pin_in_object_store, get_pinned_object
import os
from ray.tune.suggest.hyperopt import HyperOptSearch
from hyperopt import hp
import torch
class Classifier(tune.Trainable):

    def _reset_model_args(self):
        model_args = self.config['model_args'].copy()

        model_args['weight_decay'] = self.config['weight_decay']
        model_args['learning_rate'] = self.config['learning_rate']
        model_args['max_seq_length'] = int(self.config['max_seq_length'])
        model_args['adam_epsilon'] = self.config['adam_epsilon']
        model_args['warmup_ratio'] = self.config['warmup_ratio']

        self.model_args = model_args

        self.eval_df = get_pinned_object(self.config['eval_df_id'])
        self.train_df = get_pinned_object(self.config['train_df_id'])
        self.class_weights = self.config['class_weights']

    def _setup(self, config):
        print('SETTING UP')
        self.config = config

        self._reset_model_args()

        self.model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=2, weight=self.class_weights)


    def _train(self):
        self.model.train_model(self.train_df, eval_df=self.eval_df, args=self.model_args)
        results, _, _ = self.model.eval_model(self.eval_df, verbose=True)
        return results

    def _save(self, checkpoint_dir):
        print(checkpoint_dir)
        model_to_save = self.model.model.module if hasattr(self.model.model, "module") else self.model.model
        model_to_save.save_pretrained(checkpoint_dir)
        self.model.tokenizer.save_pretrained(checkpoint_dir)
        torch.save(self.model.optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer.pt'))
        torch.save(self.model.scheduler.state_dict(), os.path.join(checkpoint_dir, 'scheduler.pt'))
        return checkpoint_dir

    def _restore(self, checkpoint_path):
        self.model = ClassificationModel('distilbert', checkpoint_path, num_labels=2, weight=self.class_weights)

    def reset_config(self, new_config):
        self.config = new_config
        self._reset_model_args()
        return True

def load_df(pos_fp, neg_fp, frac=1):
    df_pos = pd.DataFrame(open(pos_fp, 'r').read().split('\n'), columns=['text'])
    df_pos['label'] = 1
    df_neg = pd.DataFrame(open(neg_fp, 'r').read().split('\n'), columns=['text'])
    df_neg['label'] = 0
    return pd.concat([df_pos, df_neg]).sample(frac=frac).reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-pos-path",
        default='datasets/constrained_classification/k16/pos.train.article.txt',
    )
    parser.add_argument(
        "--train-neg-path",
        default='datasets/constrained_classification/k16/neg.train.article.txt',
    )
    parser.add_argument(
        "--eval-pos-path",
        default='datasets/constrained_classification/k16/pos.valid.article.txt',
    )
    parser.add_argument(
        "--eval-neg-path",
        default='datasets/constrained_classification/k16/neg.valid.article.txt',
    )

    args = parser.parse_args()

    ray.init()

    train_df = load_df(args.train_pos_path, args.train_neg_path, 0.01)
    eval_df = load_df(args.eval_pos_path, args.eval_neg_path, 0.01)

    class_weights = class_weight.compute_class_weight('balanced',
                                                 [0, 1],
                                                 train_df.label)

    train_df_id = pin_in_object_store(train_df)
    eval_df_id = pin_in_object_store(eval_df)

    model_args = {
        'evaluate_during_training': True,
        'log_tune': True,
        'train_batch_size': 32,
        'gradient_accumulation_steps': 1,
        'eval_batch_size': 32,
        'num_train_epochs': 1,
        'eval_steps': 10000000,
        'save_steps': 10000000,
        'cache_dir': os.path.join(os.getcwd(), 'cache_dir'),
        'overwrite_output_dir': True
    }

    config = {
        'warmup_ratio': 0.04,
        'train_df_id': train_df_id,
        'eval_df_id': eval_df_id,
        'model_args': model_args,
        'class_weights': class_weights
    }

    space = {
        'adam_epsilon': hp.loguniform('adam_epsilon', np.log(1e-8), np.log(1e-7)),
        'weight_decay': hp.choice('weight_decay', [0, 0.01]),
        'learning_rate': hp.loguniform('learning_rate', np.log(1e-6), np.log(1e-4)),
        'max_seq_length': hp.quniform('max_seq_length', 96, 160, 1),
    }

    resources_per_trial = {
        "cpu": 8,
        "gpu": 1,
    }

    current_best_params = [
        {
            "adam_epsilon": 1e-8,
            'weight_decay': 0,
            'learning_rate': 1e-5,
            'max_seq_length': 128
        }
    ]

    algo = HyperOptSearch(
        space,
        metric="mcc",
        max_concurrent=5,
        mode="max",
        points_to_evaluate=current_best_params)

    analysis = tune.run(
        Classifier, 
        config=config, 
        search_alg=algo,
        resources_per_trial=resources_per_trial, 
        scheduler=tune.schedulers.MedianStoppingRule(time_attr='training_iteration', metric='mcc', mode='max', grace_period=3))

    print("Best config: ", analysis.get_best_config(metric='mcc'))

if __name__ == "__main__":
    main()