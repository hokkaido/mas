# MAS Data Science

## Prerequisites

The additions to 

## MASS Base-Model

Needs to be downloaded from 

https://modelrelease.blob.core.windows.net/mass/mass-base-uncased.tar.gz


## Data 

### Data acquisition

#### Conventions

There is the following convention where we put our raw downloaded datasets

    datasets/DATASET_NAME/raw

So for instance `datasets/cnndm/raw`, `datasets/xsum/raw`, `datasets/duc2004/raw` and so on.


#### CNN-DM

Here, we mainly need the acquire the following files / directories:

cnn_stories_tokenized
dm_stories_tokenized
url_lists

There are various ways to do this.

#### XSum

The official way:

https://github.com/EdinburghNLP/XSum

Please follow the instructions to download the entire data set. It will probably take quite some time and re-attempts to download all files.

However, it's much easier to download XYZ.

### Data processing

Data processing follows the following two stages

1. Clean-Up
1. Preprocessing
2. Binarization

### Clean-Up

The data sets first need to be converted and cleaned up so that they can be preprocessed and then binarized.

Unprocessed data needs to be converted into three splits:

    train.{src,tgt}
    valid.{src,tgt}
    test.{src,tgt}

where `src` denotes the source articles and `tgt` denotes the summaries.

The data can be tokenized, but doesn't have to be. More important is that the data shouldn't be lowercased in this step because the named entity recognition during the preprocessing step is very sensitive to the capitalization.

CNN-DM needs its own conversion here

    python --input-dir datasets/cnndm/raw --output-dir datasets/cnndm/preprocessed

AS does XSum needs its own conversion here

    python cleanup.py --config xsum --input-dir datasets/xsum/raw --output-dir datasets/xsum/preprocessed

This requires the splits json to be present.

As does DUC2004

    python cleanup.py --config duc2004 --input-dir datasets/duc2004/raw --output-dir datasets/duc2004/preprocessed

#### Preprocessing

Tokenizes the data, detects named entities, this will produce two folders:

* preprocessed-core
* preprocessed-entities

There is a python script that does this:

    $DATA_DIR=datasets/cnndm
    $SPLIT=train
    $ENC_OUTDIR=${DATA_DIR}/preprocessed-core
    $ENT_OUTDIR=${DATA_DIR}/preprocessed-entities

    python preprocess.py \
        --inputs ${DATA_DIR}/preprocessed/${SPLIT}.abstract.txt \
        --enc-outputs ${ENC_OUTDIR}/train.tgt \
        --ent-outputs ${ENT_OUTDIR}/${SPLIT}.tgt \
        --max-len 511 \
        --workers 40

However, it is probably easier to use or adapt one of the provided bash scripts:

    ./scripts/preprocess-cnndm.sh
    ./scripts/preprocess-duc2004.sh
    ./scripts/preprocess-xsum.sh

#### Binarization

The preprocessed data needs to be binarized so that `fairseq` can use it. This step requires a
dictionary for both `core` and `entities`. The dictionary for the the base model can be found
in it downloaded state. The dictionary for the entities can be found in `datasets/cnn

This step will produce two folders

* core
* entities

There are example scripts inside the  `scripts` folder

    ./scripts/binarize-cnndm.sh



## Approaches

### In-Domain Pretraining



### Sentence Selection

Sentence selection needs a trained sentence classifier.

#### Train classifier

See ....

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

#### fairseq

Copy generator can be enabled with

    --copy-attn

*Attention:* FP16 is not supported here.

An example can be found at

    ./scripts/examples/train-with-copy.sh

### In-Domain Pretraining

## Hyperparameter optimization

`gen-hpb-search.py` can be used to optimize the decoding parameters. At the moment, the following arguments (in addition to the training arguments) are supported.

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

The configuration spaces are currently hard-coded in side the python file.

There are three example scripts

Start host

    ./tune-hpb-inference-cnndm.sh

Run worker in new tab
    ./tune-hpb-inference-cnndm.sh --hpb_worker

Run worker in new tab
    ./tune-hpb-inference-cnndm.sh --hpb_worker

Visualize results


## Evaluation

Generate hypotheses on the test sets and calculate rouge rcores for a checkpoint and multiple configs

    ./eval-all-cnndm.sh

Generate BWS-Samples:

    python bws-eval.py

Evaluate MTurk Evaluations:

    python mturk-eval.py

Create latex table rows:

    python parse-rouge.py

Textual complexity

Consult notebook
