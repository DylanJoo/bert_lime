#!/bin/sh
# python3 test.py \
#   --output_dir test \
#   --model_name_or_path 'textattack/bert-base-uncased-SST-2' \
#   --tokenizer_name 'textattack/bert-base-uncased-SST-2' \
#     > log.out 

python3 test.py \
  --output_dir test \
  --model_name_or_path 'textattack/bert-base-uncased-snli' \
  --tokenizer_name 'textattack/bert-base-uncased-snli' \
    > log.out 
