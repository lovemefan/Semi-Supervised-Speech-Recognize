#!/bin/bash
arr=("train" "test" "dev")
for item in "${arr[@]}"
  do
    python convert_pinyin.py  --tsv /root/dataset/speechDataset/aishell-1/data_aishell/transcript/$item.tsv \
    --tsv_output /root/dataset/speechDataset/aishell-1/data_aishell/transcript
  done


# build vocab file
python generate_vocabulary.py --tsv /root/dataset/speechDataset/aishell-1/resource_aishell/manifests_all.tsv \
      --vocab_output /root/dataset/speechDataset/aishell-1/resource_aishell/dict.txt