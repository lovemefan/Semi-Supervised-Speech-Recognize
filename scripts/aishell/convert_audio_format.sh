#!/bin/bash
arr=("train" "test" "dev")
for item in "${arr[@]}"
  do
    python convert_audio.py  --tsv /root/dataset/speechDataset/aishell-1/data_aishell/transcript/$item.tsv \
    --audio_output /root/dataset/speechDataset/aishell-1/data_aishell/converted_wav/$item \
    --tsv_output /root/dataset/speechDataset/aishell-1/data_aishell/transcript
  done