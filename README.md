# Transfer Learning for 📝 Text Summarization

Presented approaches

* Embedding named entities
* Embedding segment positions
* Adding copy-mechanism
* Sentence selection
* Masked in-domain pretraining (improved)

## Prerequisites

The repository has been tested with Ubuntu 18.04 and CUDA 10.2. It requires Anaconda, git, NVIDIA Apex, wget and unzip to work.

Create the conda environment and install immediate dependencies

    conda env create -f environment.yml

Initialize git submodules

    git submodule update --init

## Project structure

| Folder      | Description |
| ----------- | ----------- |
| .             | Some CLI commands live here. |
| checkpoints   | Place where fairseq training checkpoints are stored.  |
| datasets      | Location of data sets.  |
| eval           | Location for evaluation results.  |
| deps          | This contains our model additions to MASS, which was forked from the original implementation.   |
| mas           | Various python modules.  |
| notebooks           | Several notebooks, especially for evaluation.  |
| scripts           | Bash scripts for training, data processing, evaluation and so on. |

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

These raw data sets then have to be preprocessed and binarized (see below).

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

Data processing follows the following stages

1. Clean-Up
1. Preprocessing
2. Binarization

### Clean-Up

The data sets first need to be converted and cleaned up so that they can be preprocessed and then binarized.

Unprocessed raw data needs to be converted into the source target pairs per split, depending on the data set this will produce files like this:

    train.{src,tgt}
    valid.{src,tgt}
    test.{src,tgt}

where `src` denotes the source articles and `tgt` denotes the summaries.

The data can be tokenized during this step, but doesn't have to be. More important is that the data shouldn't be lowercased in this step because the named entity recognition during the preprocessing step is very sensitive to the capitalization.

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
dictionary for both `core` and `entities`. The dictionary for the base MASS model can be found
in its archive, but it is also provided together with the entities dictionary at `datasets/shared`.

The binarization step will produce two folders

* core
* entities

There are example scripts inside the `scripts` folder that can be used for the binarization:

    ./scripts/data/binarize-cnndm.sh
    ./scripts/data/binarize-duc2004.sh
    ./scripts/data/binarize-xsum.sh

## Training

You can find examples for the training and fine-tuning scripts under `scripts/examples` and even more inside `scripts/other`. Suffice to say that especially the batch size, learning rate and gradient accumulation steps will depend on your setup. 

fairseq needs to be informed of the model extensions by setting a value for `--user-dir` that points to the git sub-module at `--user-dir deps/MASS/MASS-summarization/mass` and settings the `--arch` flag to `summarization_mass_base`.

There are two different learning tasks `augmented_summarization_mass`, which is used for fine-tuning with additional embedding layers and the copy-mechanism and `masked_summarization_mass` which enables masked in-domain pretraining.

An example for a complete fine-tuning command without using any of the addtional embedding layers is shown below:

    fairseq-train datasets/cnndm/ \
        --user-dir deps/MASS/MASS-summarization/mass --task augmented_summarization_mass --arch summarization_mass_base \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 0.0005 --min-lr 1e-09 \
        --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
        --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --update-freq 8 --max-tokens 4096 \
        --ddp-backend=no_c10d --max-epoch 25 \
        --max-source-positions 512 --max-target-positions 512 \
        --fp16 \
        --memory-efficient-fp16 \
        --skip-invalid-size-inputs-valid-test \
        --load-from-pretrained-model checkpoints/mass-base-uncased/mass-base-uncased.pt \

### Embedding Entities

Entities can be embedded on both the encoder side with the following args

    [--embed-entities-encoder]
    [--embed-entities-decoder]

| Argument      | Description |
| ----------- | ----------- |
| --embed-entities-encoder     | Use an entities embedding layer with the encoder. Default: `false`  |
| --embed-entities-decoder    | Use an entities embedding layer with the decoder. Default: `false`  |

An example can be found at

    ./scripts/examples/fine-tune-with-entities.sh

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

    ./scripts/examples/fine-tune-with-segments.sh


### Copy-mechanism

The copy mechanism can be enabled with

    --copy-attn

You also need to use a different criterion here:

    --criterion copy_generator_loss

*Attention:* FP16 is not supported here. 

An example can be found at

    ./scripts/examples/fine-tune-with-copy.sh


### Sentence Selection

Sentence selection needs two things:

* A constrained training and validation data set
* A trained sentence classifier that can be applied on the test set

After that, the constrained data set can be fine-tuned and evaluated as normal. Unfortunately, the procedure to constrain the data sets is a bit involved. 

#### Data preparation

There is an example script that constrains the training and validation sets of CNN-DM and creates class labels for the sentence classifier:

    ./scripts/examples/constrain-cnndm.sh

This script will take a while, the constrained files and labels will be created `datasets/cnndm-constrained/preprocessed` and `datasets/cnndm/labels` respectively.

You could now preprocess this constrained data to create the core augmented data sets if you want to use the named entities embedding layer:

    ./scripts/data/preprocess-constrained-cnndm.sh

Ideally, we only want to run this step once, for the training and validation sets. Once preprocessed, we can fine-tune a model with it and then evaluate it on multiple constrained test sets if you are experimenting with multiple sentence classifiers. 

#### Train classifier

In order to constain the test sets, we need to train a sentence classifier on the labels which were extracted in the previous step.

We have experimented with multiple sentence classifiers, an example is provided for training one for CNN-DM and based on XLNet that you can adapt further:

    ./examples/train-selector.sh

This will train a classifier and periodically save a checkpoint to /output.

You can look at the progress with tensorboard

    ./tensorboard --logdir tensorboard

#### Use classifier to constrain test set

After you have trained a classifier, you need to apply it to the test set. An example for this can be found at:

    ./scripts/examples/constrain-with-trained-selector.sh 

Please adapt it to make use of the correct checkpoint.

After the test set has been constrained, we need to preprocess and binarize the complete CNN-DM dataset, you can use or adapt

    ./scripts/data/preprocess-constrained-cnndm-test.sh
    ./scripts/data/binarize-constrained.sh

#### Fine-Tune

Now you can fine-tune and evaluate the constrained data set by adapting any of the example scripts.

### Masked In-Domain Pretraining

A variant of the masked in-domain pretraining as described in the thesis is presented here, it provides better results than the originally proposed scheme, but it is still not better than the normal fine-tuning procedure of MASS. In this variant, a joint loss is optimized during pretraining instead of single loss. It is inspired by the supervised neural machine translation version of MASS.

The originally proposed masking scheme is virtually unchanged, it still contains the sentence permutation of the source text, and masks a portion of the overlapping tokens between source and summary, it also also randomly replaces certain masked tokens with a random word from the vocabulary. However, it doesn't randomly flip between masking overlapping and non-overlapping tokens, and the goal here is to predict the original source text.

Jointly, we also try to predict a partially corrupted (masked) version of the entire target sequence, not just the masked overlapping tokens. Both of these losses are added together during the training step.

First, we pretrain in-domain until the validation loss stops decreasing:

    ./scripts/examples/pretrain-cnndm-joint.sh

And then we fine-tune as with the other

    ./scripts/examples/fine-tune-on-pretrained-joint.sh

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

Visualize the optimization results during run by pointing to the run_id:

    python visualize_bohb.py --run_id runs/cnndm-vanilla

## Evaluation

### Super simple evaluation

If you just want to evaluate a single model, you can adapt the following script:

    ./scripts/examples/infer-cnndm.sh

This saves the output to `output.txt`. You can now calculate the ROUGE-scores with:

    ./scripts/eval/rouge.sh

### Evaluating multiple models at once

We recommend using the `eval` directory for evaluating multiple models at once. For instance, let's say you have trained four models that reside inside the following folders:

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

Then, set the parameters that you found by overriding `BEAM_PARAMS`:

    BEAM_PARAMS="--beam 3 --min-len 45 --no-repeat-ngram-size 3"

Now, set the models and set the correct fairseq parameters

    MODELS=(
        ['cnndm-vanilla']=''
        ['cnndm-segments-encoder']="--embed-segments-encoder --segment-tokens . --max-segments 128"
        ['cnndm-entities-encoder']="--embed-entities-encoder"
    )

Running `./scripts/eval/eval-all-cnndm.sh` will evaluate the three models with ROUGE on the test set and save the results inside the `eval` dir. 

### Further analysis

Please consult the `notebooks` directory for some ideas on how to evaluate the output of the models even further.

### Human Evaluation

You can also create BestWorstScale samples for an MTurk experiment:

    python create_bws_eval.py --experiment-dir eval/exp1-cnndm --models cnndm-reference cnndm-entities-encoder cnndm-segments-encoder

After you have done a human MTurk Evaluation with the generated CSV, you can adapt `mturk_eval.py` to your liking to calculate the BWS-score.


## Acknowledgements

This repository contains code that is an extension of or is inspired from:

https://github.com/microsoft/MASS (Base Architecture)
https://github.com/OpenNMT/OpenNMT-py (Copy Mechanism)
https://github.com/ThilinaRajapakse/simpletransformers/ (Sentence Classification)
