from mas import create_bws_eval

if __name__ == "__main__":
    exp_path='eval/exp2-xsum'
    num_evaluators=3
    models_to_eval=['xsum-vanilla', 'xsum-entities-encoder', 'xsum-entities-encoder-segments-encoder']
    create_bws_eval(exp_path=exp_path, num_evaluators=num_evaluators, models_to_eval=models_to_eval, seed=2020)