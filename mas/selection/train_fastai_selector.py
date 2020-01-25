import argparse
from fastai.text import * 
import pandas as pd
from sklearn.utils import class_weight

def load_df(pos_fp, neg_fp, frac=1):
    df_pos = pd.DataFrame(open(pos_fp, 'r').read().split('\n'), columns=['text'])
    df_pos['label'] = 1
    df_neg = pd.DataFrame(open(neg_fp, 'r').read().split('\n'), columns=['text'])
    df_neg['label'] = 0
    return pd.concat([df_pos, df_neg]).sample(frac=frac).reset_index(drop=True)

def convert(args):
    train_df = load_df(args.train_pos_path, args.train_neg_path, 0.1)
    eval_df = load_df(args.eval_pos_path, args.eval_neg_path, 0.1)

    data_lm = TextLMDataBunch.from_df(path='.', train_df=train_df, valid_df=eval_df, bs=32)
   
    data_clas = TextClasDataBunch.from_df(path='.', train_df=train_df, valid_df=eval_df,  vocab=data_lm.train_ds.vocab, bs=32)
    data_lm.save('data_lm_export.pkl')
    data_clas.save('data_clas_export.pkl')

def pretrain():
    #train_df = load_df(args.train_pos_path, args.train_neg_path, 0.5)
    #eval_df = load_df(args.eval_pos_path, args.eval_neg_path, 0.5)

    #data_lm = TextLMDataBunch.from_df(path='.', train_df=train_df, valid_df=eval_df)
   
    #data_clas = TextClasDataBunch.from_df(path='.', train_df=train_df, valid_df=eval_df,  vocab=data_lm.train_ds.vocab, bs=32)
    #data_lm.save('data_lm_export.pkl')
    #data_clas.save('data_clas_export.pkl')
    data_lm = load_data('.', 'data_lm_export.pkl', bs=32)
    data_clas = load_data('.', 'data_clas_export.pkl', bs=32)

    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5).to_fp16()

    print('finding learning rate')
    learn.lr_find() # find learning rate
    fig = learn.recorder.plot(suggestion=True, return_fig=True)
    fig.savefig('lr.png')
    learn.fit_one_cycle(1, learn.recorder.min_grad_lr)
    learn.unfreeze() # must be done before calling lr_find
    learn.lr_find()
    fig = learn.recorder.plot(suggestion=True, return_fig=True)
    fig.savefig('lr2.png')
    learn.fit_one_cycle(10, learn.recorder.min_grad_lr)
    learn.save_encoder('k16-enc')

def finetune():
    data_lm = load_data('.', 'data_lm_export.pkl', bs=32)
    data_clas = load_data('.', 'data_clas_export.pkl', bs=32)

    learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5).to_fp16()
    learn.load_encoder('k16-enc')

    learn.lr_find()
    learn.recorder.plot(suggestion=True)
    min_grad_lr = learn.recorder.min_grad_lr
    learn.fit_one_cycle(2, min_grad_lr)
    fig = learn.recorder.plot_losses(return_fig=True)
    fig.savefig('losses.png')
    learn.freeze_to(-2)
    learn.fit_one_cycle(4, slice(5e-3, 2e-3), moms=(0.8,0.7))
    fig = learn.recorder.plot_losses(return_fig=True)
    fig.savefig('losses2.png')
    learn.unfreeze()
    learn.fit_one_cycle(4, slice(2e-3/100, 2e-3), moms=(0.8,0.7))
    fig = learn.recorder.plot_losses(return_fig=True)
    fig.savefig('losses3.png')
    learn.export()

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

    #train_df = load_df(args.train_pos_path, args.train_neg_path, 0.5)
    #eval_df = load_df(args.eval_pos_path, args.eval_neg_path, 0.5)

    #data_lm = TextLMDataBunch.from_df(path='.', train_df=train_df, valid_df=eval_df)
   
    #data_clas = TextClasDataBunch.from_df(path='.', train_df=train_df, valid_df=eval_df,  vocab=data_lm.train_ds.vocab, bs=32)
    #data_lm.save('data_lm_export.pkl')
    #data_clas.save('data_clas_export.pkl')
    
    #convert(args)
    #pretrain()
    #finetune()


    learn = load_learner('.', file='export.pkl', bs=32)


    log_preds = learn.get_preds(ds_type=DatasetType.Valid, ordered=True)
    print(log_preds[1])
if __name__ == "__main__":
    main()