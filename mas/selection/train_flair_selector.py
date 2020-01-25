from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, RoBERTaEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.data import Dictionary
from pathlib import Path
from flair.training_utils import (
    EvaluationMetric,
)
from flair.data import Corpus
from flair.datasets import ClassificationCorpus

from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter,TextClassifierParamSelector, OptimizationValue

def optimize():
    corpus, label_dictionary = load_corpus()
    corpus.downsample(0.01)
    # define your search space
    search_space = SearchSpace()
    #embeddigns[  RoBERTaEmbeddings(pretrained_model_name_or_path="roberta-base", layers="0,1,2,3,4,5,6,7,8,9,10,11,12",
                                #pooling_operation="first", use_scalar_mix=True) ]
    embeddings = [WordEmbeddings('glove'), FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')]
    search_space.add(Parameter.EMBEDDINGS, hp.choice, options=[embeddings])
    search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128, 256, 512])
    search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
    search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
    search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
    search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[16, 32, 64])



# create the parameter selector
    param_selector = TextClassifierParamSelector(
        corpus, 
        False, 
        'resources/results', 
        'lstm',
        max_epochs=10,
        training_runs=3,
        optimization_value=OptimizationValue.DEV_SCORE,
        label_dictionary=label_dictionary
    )

# start the optimization
    param_selector.optimize(search_space, max_evals=100)

def load_corpus():
    label_dictionary: Dictionary = Dictionary(add_unk=False)
    label_dictionary.multi_label = False

    label_dictionary.add_item('0')
    label_dictionary.add_item('1')

    # this is the folder in which train, test and dev files reside
    data_folder = 'datasets/constrained_classification/k16'

    # load corpus containing training, test and dev data
    corpus: Corpus = ClassificationCorpus(data_folder,
                                        dev_file='fasttext.valid',
                                        train_file='fasttext.train')

    return corpus, label_dictionary

def optimize_lr():

    corpus, label_dictionary = load_corpus()

    embeddings =[WordEmbeddings('glove'), FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')]

    document_embeddings = DocumentRNNEmbeddings(embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256, bidirectional=True)
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dictionary, multi_label=False)
    trainer = ModelTrainer(classifier, corpus)

    # 7. find learning rate
    learning_rate_tsv = trainer.find_learning_rate('resources/classifiers/',
                                                        'learning_rate.tsv')

    # 8. plot the learning rate finder curve
    from flair.visual.training_curves import Plotter
    plotter = Plotter()
    plotter.plot_learning_rate(learning_rate_tsv)

def train():

    corpus, label_dictionary = load_corpus()


    embeddings =[WordEmbeddings('glove'), FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')]

    document_embeddings = DocumentRNNEmbeddings(embeddings, hidden_size=512, reproject_words=True, reproject_words_dimension=256, bidirectional=True)
    classifier = TextClassifier(document_embeddings, label_dictionary=label_dictionary, multi_label=False)
    trainer = ModelTrainer(classifier, corpus)
    trainer.train('flair_checkpoints', max_epochs=10, use_amp=True, learning_rate=9e-02, mini_batch_size=32)

if __name__ == "__main__":
    train()

