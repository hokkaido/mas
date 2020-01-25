#!/bin/bash

DATA_DIR=datasets

mkdir -p ${DATA_DIR}/cnndm/raw
echo "Downloading CNNDM"
git clone https://github.com/becxer/cnn-dailymail.git ${DATA_DIR}/cnndm/raw
wget --no-check-certificate "https://onedrive.live.com/download?cid=B52BFC9F974BE0BC&resid=B52BFC9F974BE0BC%211632&authkey=AMmXm18luci9bo8" -O ${DATA_DIR}/cnndm/raw/cnn_stories_tokenized.zip  -q --show-progress
wget --no-check-certificate "https://onedrive.live.com/download?cid=B52BFC9F974BE0BC&resid=B52BFC9F974BE0BC%211631&authkey=APjAuMnc7VOP78I" -O ${DATA_DIR}/cnndm/raw/dm_stories_tokenized.zip  -q --show-progress
unzip -qq ${DATA_DIR}/cnndm/raw/dm_stories_tokenized.zip  -d ${DATA_DIR}/cnndm/raw
unzip -qq ${DATA_DIR}/cnndm/raw/cnn_stories_tokenized.zip -d ${DATA_DIR}/cnndm/raw

mkdir -p ${DATA_DIR}/xsum/raw
echo "Downloading XSum"
wget https://github.com/EdinburghNLP/XSum/raw/master/XSum-Dataset/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json -O ${DATA_DIR}/xsum/raw/XSum-TRAINING-DEV-TEST-SPLIT-90-5-5.json  -q --show-progress
wget http://kinloch.inf.ed.ac.uk/public/XSUM-EMNLP18-Summary-Data-Original.tar.gz -O ${DATA_DIR}/xsum/raw/XSUM-EMNLP18-Summary-Data-Original.tar.gz  -q --show-progress
tar -xzf ${DATA_DIR}/xsum/raw/XSUM-EMNLP18-Summary-Data-Original.tar.gz -C ${DATA_DIR}/xsum/raw

mkdir -p ${DATA_DIR}/duc2004/raw
echo "Downloading DUC2004"
git clone https://github.com/echalkpad/snickebod.git ${DATA_DIR}/duc2004/raw/snickebod
mv ${DATA_DIR}/duc2004/raw/snickebod/duc2004/* ${DATA_DIR}/duc2004/raw/


