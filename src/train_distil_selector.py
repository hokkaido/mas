import argparse
import transformers
from classification_model import ClassificationModel

import pandas as pd
from sklearn.utils import class_weight

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

    train_df = load_df(args.train_pos_path, args.train_neg_path, 0.25)
    eval_df = load_df(args.eval_pos_path, args.eval_neg_path, 0.25)

    model_args = {
        'evaluate_during_training': True,
        'train_batch_size': 32,
        'gradient_accumulation_steps': 8,
        'eval_batch_size': 32,
        'num_train_epochs': 1,
        'logging_steps': 50,
        'save_steps': 500,
        'eval_steps': 100,
        'max_seq_length': 128,
        'learning_rate': 1e-5,
        'warmup_ratio': 0.04,
    }

    target_count = train_df.label.value_counts()
    print('Class 0:', target_count[0])
    print('Class 1:', target_count[1])

    class_weights = class_weight.compute_class_weight('balanced',
                                                 [0, 1],
                                                 train_df.label)
   

    model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=2, weight=class_weights)

    model.train_model(train_df, eval_df=eval_df, args=model_args)
    
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)

    #print(result)
    
    #python src/train_selector.py 

if __name__ == "__main__":
    main()