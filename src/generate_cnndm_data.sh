# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Ensure the output directory exists
data_dir=datasets/cnndm
para_data_dir=$data_dir/processed
save_dir=$data_dir/fqprocessed

# set this relative path of MASS in your server
user_dir=deps/MASS/MASS-fairseq/mass

mkdir -p $data_dir $save_dir $para_data_dir


# Generate Bilingual Data
fairseq-preprocess \
  --user-dir $user_dir \
  --task xmasked_seq2seq \
  --source-lang ar --target-lang ab \
  --trainpref $para_data_dir/train --validpref $para_data_dir/valid --testpref $para_data_dir/test \
  --destdir $save_dir \
  --fp16

