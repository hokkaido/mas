import argparse
import transformers
from .classification_model import ClassificationModel

import pandas as pd
from sklearn.utils import class_weight

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

def train_xl_selector(args):
    train_df = load_df(args.train_path, args.sample)
    eval_df = load_df(args.eval_path, args.sample)

    model_args = {
        'evaluate_during_training': True,
        'train_batch_size': 32,
        'gradient_accumulation_steps': 1,
        'eval_batch_size': 32,
        'num_train_epochs': 10,
        'logging_steps': 100,
        'save_steps': 500, # attn scales with accumulation steps
        'eval_steps': 250,
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
    
    model = ClassificationModel('xlnet', 'xlnet-base-cased', num_labels=2, weight=class_weights)

    model.train_model(train_df, eval_df=eval_df, args=model_args)
    
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)

    #print(result)
    
    #python src/train_selector.py 

