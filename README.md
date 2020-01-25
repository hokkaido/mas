# Transfer Learning for 📝 Text Summarization

## Prerequisites

The repository has been tested with Ubuntu 18.04 and CUDA 10.2. It requires Anaconda, git, wget and unzip to work.

Create the conda environment and install immediate dependencies

    conda env create -f environment.yml

Initialize git submodules

    git submodule update --init

## Project structure

| Folder      | Description |
| ----------- | ----------- |
| .             | Some CLI commands live here |
| checkpoints   | Place where training checkpoints are stored.  |
| datasets      | Location of data sets  |
| eval           | Location for evaluation results  |
| deps          | This contains our model additions to MASS, which was forked from the original   |
| mas           | Various python modules  |
| notebooks           | Several notebooks, especially for evaluation  |
| scripts           | Training and so on   |

## MASS Base-Model

The pretrained base model can be downloaded from [here](https://modelrelease.blob.core.windows.net/mass/mass-base-uncased.tar.gz).

You can download it automatically with

    ./scripts/models/download.sh

This will create download and extract the base model to `checkpoints/mass-base-uncased`

## Data 

We currently support three data sets, `cnndm`, `xsum` and `duc2004`.

There is a script that downloads the three datasets where you don't have to follow the manual steps below.

    ./scripts/data/download.sh

This will create three folders, `datasets/cnndm/raw`, `datasets/xsum/raw`, `datasets/duc2004/raw`.

These raw data sets then have to be preprocessed and binarized.

### Data acquisition

If you don't want to rely on the bash script above, you can also do this by following the

#### CNN-DM

Here, we mainly need the acquire the following files / directories:

cnn_stories_tokenized
dm_stories_tokenized
url_lists

There are various ways to do this, with following the instructions at https://github.com/abisee/cnn-dailymail being one option.

#### XSum

There is the original [XSum repository](https://github.com/EdinburghNLP/XSum) with instructions. Please follow the instructions to download the entire data set. It will probably take quite some time and re-attempts to download all files.

### Duc2004

Please contact [NIST](https://duc.nist.gov/duc2003/tasks.html) directly.

### Data processing

Data processing follows the following two stages

1. Clean-Up
1. Preprocessing
2. Binarization

### Clean-Up

The data sets first need to be converted and cleaned up so that they can be preprocessed and then binarized.

Unprocessed data needs to be converted into the source target pairs per split, depending on the data set this will produce files like this:

    train.{src,tgt}
    valid.{src,tgt}
    test.{src,tgt}

where `src` denotes the source articles and `tgt` denotes the summaries.

The data can be tokenized, but doesn't have to be. More important is that the data shouldn't be lowercased in this step because the named entity recognition during the preprocessing step is very sensitive to the capitalization.

There is a script that does this automatically for all three datasets:

    ./scripts/data/cleanup.sh

However, you can also do this individually:

    python cleanup.py --config cnndm --input-dir datasets/cnndm/raw --output-dir datasets/cnndm/preprocessed
    python cleanup.py --config xsum --input-dir datasets/xsum/raw --output-dir datasets/xsum/preprocessed
    python cleanup.py --config duc2004 --input-dir datasets/duc2004/raw --output-dir datasets/duc2004/preprocessed

#### Preprocessing

Tokenizes the data, detects named entities, limits the document lengths, this will produce two folders:

* preprocessed-core
* preprocessed-entities

We have also provided individual bash scripts that should work out of the box if you have used the download and cleanup scripts above. They limit the output to 511 tokens per article or summary. **Warning**: This is a slow process (**hours**, not minutes), mainly due to the entity recognition step. If you run out of memory, try lowering the workers inside the scripts.

    ./scripts/data/preprocess-cnndm.sh
    ./scripts/data/preprocess-duc2004.sh
    ./scripts/data/preprocess-xsum.sh

There is also a python script that can be used for this:

    DATA_DIR=datasets/cnndm
    SPLIT=train
    ENC_OUTDIR=${DATA_DIR}/preprocessed-core
    ENT_OUTDIR=${DATA_DIR}/preprocessed-entities

    python preprocess.py \
        --inputs ${DATA_DIR}/preprocessed/${SPLIT}.abstract.txt \
        --enc-outputs ${ENC_OUTDIR}/train.tgt \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.tgt \
        --max-len 511 \
        --workers 40

#### Binarization

The preprocessed data needs to be binarized so that `fairseq` can use it. This step requires a
dictionary for both `core` and `entities`. The dictionary for the the base model can be found
in it downloaded state. The dictionary for the entities can be found in `datasets/cnn

This step will produce two folders

* core
* entities

There are example scripts inside the `scripts` folder.

    ./scripts/data/binarize-cnndm.sh
    ./scripts/data/binarize-duc2004.sh
    ./scripts/data/binarize-xsum.sh

## Approaches

### Simple-Finetuning



### In-Domain Pretraining




### Sentence Selection

Sentence selection needs two things:

* A constrained train and validation data set
* A trained sentence classifier


#### Data preparation

There is an example script that constrains CNN-DM, and creates class labels that can be adapted:

    ./scripts/examples/constrain-cnndm.sh

This will take a while and create files inside `datasets/cnndm-constrained/preprocessed` and `datasets/cnndm/labels`.

After this step, we need to preprocess and binarize this data, you can use or adapt

    ./scripts/data/preprocess-constrained-cnndm.sh
    ./scripts/data/binarize-constrained.sh

#### Train classifier

We have experimented with multiple classifiers, an example is provided for training one based on XLNet:

    ./examples/train-selector.sh

This will train a classifier and periodically check

You can look at the progress with tensorboar

#### Use classifier to constrain test set



#### Constrain data

See ...

## Training (fairseq)

### Embedding Entities

Entities can be embedded on both the encoder side with the following args

    [--embed-entities-encoder]
    [--embed-entities-decoder]

| Argument      | Description |
| ----------- | ----------- |
| --embed-entities-encoder     | Use an entities embedding layer with the encoder. Default: `false`  |
| --embed-entities-decoder    | Use an entities embedding layer with the decoder. Default: `false`  |

An example can be found at

    ./scripts/examples/finetune-with-entities.sh

### Embedding Segments

There are four arguments that control the embedding of segment positions:

    [--embed-segments-encoder]
    [--embed-segments-decoder]
    [--segment-tokens SEGMENT_TOKENS]
    [--max-segments MAX_SEGMENTS]

| Argument      | Description |
| ----------- | ----------- |
| --embed-segments-encoder     | Use a segment embedding layer with the encoder. Default: `false`  |
| --embed-segments-decoder     | Use a segment embedding layer with the decoder. Default: `false`  |
| --segment-tokens     | Comma separated string, tokens used for segmentation. Default: `None`. Example: `".,!,?"`  |
| --max-segments     | Maximum amount of segments. Default: `64`  |

An example can be found at

    ./scripts/examples/finetune-with-segments.sh


### Sentence Selection

Sentence selection requires a pre-constrained data set.

### Copy generator

Copy generator can be enabled with

    --copy-attn

*Attention:* FP16 is not supported here.

An example can be found at

    ./scripts/examples/finetune-with-copy.sh

### In-Domain Pretraining

## Hyperparameter optimization

`run_bohb.py` can be used to optimize the decoding parameters. At the moment, the following arguments (in addition to the training arguments) are supported.

    [--hpb_config CONFIG]
    [--hpb_runs_dir RUNS_DIR]
    [--hpb_run_id RUN_ID]
    [--hpb_min_budget MIN_BUDGET]
    [--hpb_max_budget MAX_BUDGET]
    [--hpb_metric METRIC]
    [--hpb_n_iterations N_ITERATIONS]
    [--hpb_n_workers N_WORKERS]
    [--hpb_overwrite_run]
    [--hbp_worker]


| Argument      | Description |
| ----------- | ----------- |
| --hpb_config     | `cnndm` or `xsum` |
| --hpb_run_id   | Name of the run, will be saved to `RUNS_DIR/RUN_ID`|
| --hpb_min_budget   | Minimal training budget, default `1`|
| --hpb_max_budget   | Maximal training budget, default `32`|
| --hpb_n_iterations   | Iterations to run, default `32`|
| --hpb_n_workers   | How many worker processes are required, default `2`|
| --hpb_metric   | Name of the metric, default `ROUGE-1-F (avg)`|
| --hpb_overwrite_run   | Overwrite run data if present? Default: false |
| --hpb_runs_dir  | Location where runs will be saved to, default: `runs`|
| --hpb_worker  | Run in worker mode? Default: false (server mode) |

The configuration spaces are currently hard-coded inside the python file.

There are three example scripts under `scripts/optimize` that can be adapted.

Start host

    ./scripts/optimize/tune-hpb-inference-cnndm.sh

Run worker 1 in new tab

    ./scripts/optimize/tune-hpb-inference-cnndm.sh --hpb_worker

Run worker 2 in new tab

    ./scripts/optimize/tune-hpb-inference-cnndm.sh --hpb_worker

Visualize results during run by pointing to the run_id:

    python visualize_bohb.py --run_id runs/cnndm-vanilla

## Evaluation

We recommend using the `eval` directory for this. For instance, let's say you have trained four models that reside inside the following folders:

- checkpoints/cnndm-vanilla
- checkpoints/cnndm-entities-encoder
- checkpoints/cnndm-segments-encoder

And you have used the `BOHB` hyper-parameter script and found a certain parameter combination which is best for cnndm-vanilla.

The first step would be to adapt the following script to include the four models and the parameter combination:

    ./scripts/eval/eval-all-cnndm.sh

Let's use it to create an experiment evaluation, first, change the following line to a name that you prefer for your experiment:

    EXPERIMENT=exp1-cnndm

The checkpoints dir already looks good:

    MODELDIR=checkpoints

Then, set the parameters that you found:

    BEAM_PARAMS="--beam 3 --min-len 45 --no-repeat-ngram-size 3"

Now, set the models and set the correct fairseq parameters

    MODELS=(
        ['cnndm-vanilla']=''
        ['cnndm-segments-encoder']="--embed-segments-encoder --segment-tokens . --max-segments 128"
        ['cnndm-entities-encoder']="--embed-entities-encoder"
    )


Running `./scripts/eval/eval-all-cnndm.sh` will evaluate the three models with ROUGE on the test set and save the results inside the `eval` dir.

### Further Analysis

Please consult the `notebooks` directory for some ideas on how to evaluate these results.

### Human Evaluation

You can also create BestWorstScale samples for an MTurk experiment:

    python create_bws_eval.py --experiment-dir "eval/exp1-cnndm" --models cnndm-reference cnndm-entities-encoder cnndm-segments-encoder

After you have done a human MTurk Evaluation with the, you can adapt `mturk_eval.py` to your liking to calculate the BWS-score.

Create latex table rows:

    python pprint_rouge.py
