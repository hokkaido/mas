# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

CNNDMPATH=datasets/cnndm/finished_files

CODES=40000

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --cnndmpath)
	CNNDMPATH="$2"; shift 2;;
  --reload_codes)
	RELOAD_CODES="$2"; shift 2;;
  --reload_vocab)
	RELOAD_VOCAB="$2"; shift 2;;
  --replace_ner)
	REPLACE_NER="$2"; shift 2;;
  --replace_unk)
	REPLACE_UNK="$2"; shift 2;;
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

# Check parameters

if [ "$RELOAD_CODES" != "" ] && [ ! -f "$RELOAD_CODES" ]; then echo "cannot locate BPE codes"; exit; fi
if [ "$RELOAD_VOCAB" != "" ] && [ ! -f "$RELOAD_VOCAB" ]; then echo "cannot locate vocabulary"; exit; fi
if [ "$RELOAD_CODES" == "" -a "$RELOAD_VOCAB" != "" -o "$RELOAD_CODES" != "" -a "$RELOAD_VOCAB" == "" ]; then echo "BPE codes should be provided if and only if vocabulary is also provided"; exit; fi

# main paths
MAIN_PATH=$PWD
TOOLS_PATH=$PWD/deps
DATA_PATH=$PWD/datasets/cnndm
PROC_PATH=$DATA_PATH/processed/

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $PROC_PATH

TRAIN_SRC=$CNNDMPATH/train.article.txt
TRAIN_TGT=$CNNDMPATH/train.abstract.txt
VALID_SRC=$CNNDMPATH/valid.article.txt
VALID_TGT=$CNNDMPATH/valid.abstract.txt
TEST_SRC=$CNNDMPATH/test.article.txt
TEST_TGT=$CNNDMPATH/test.abstract.txt

TRAIN_SRC_BPE=$PROC_PATH/train.ar
TRAIN_TGT_BPE=$PROC_PATH/train.ab
VALID_SRC_BPE=$PROC_PATH/valid.ar
VALID_TGT_BPE=$PROC_PATH/valid.ab
TEST_SRC_BPE=$PROC_PATH/test.ar
TEST_TGT_BPE=$PROC_PATH/test.ab

BPE_CODES=$PROC_PATH/codes
FULL_VOCAB=$PROC_PATH/vocab.ab-ar

if [ ! -f "$BPE_CODES" ] && [ -f "$RELOAD_CODES" ]; then
  echo "Reloading BPE codes from $RELOAD_CODES ..."
  cp $RELOAD_CODES $BPE_CODES
fi

# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $CODES $TRAIN_SRC $TRAIN_TGT > $BPE_CODES
fi
echo "BPE learned in $BPE_CODES"

if [ ! -f "$FULL_VOCAB" ] && [ -f "$RELOAD_VOCAB" ]; then
  echo "Reloading vocabulary from $RELOAD_VOCAB ..."
  cp $RELOAD_VOCAB $FULL_VOCAB
fi

if [ ! -f "$TRAIN_SRC_BPE" ]; then
  echo "Applying article BPE codes..."
  $FASTBPE applybpe $TRAIN_SRC_BPE $TRAIN_SRC $BPE_CODES
fi

if [ ! -f "$TRAIN_TGT_BPE" ]; then
  echo "Applying title BPE codes..."
  $FASTBPE applybpe $TRAIN_TGT_BPE $TRAIN_TGT $BPE_CODES
fi

# extract full vocabulary
if ! [[ -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $TRAIN_SRC_BPE $TRAIN_TGT_BPE > $FULL_VOCAB
fi
echo "Full vocab in: $FULL_VOCAB"

$FASTBPE applybpe $VALID_SRC_BPE $VALID_SRC $BPE_CODES 
$FASTBPE applybpe $VALID_TGT_BPE $VALID_TGT $BPE_CODES 
$FASTBPE applybpe $TEST_SRC_BPE $TEST_SRC $BPE_CODES 
$FASTBPE applybpe $TEST_TGT_BPE $TEST_TGT $BPE_CODES 