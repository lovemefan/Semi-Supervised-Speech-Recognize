python utils/infer.py /home/dataset/wenet_s/wav2vec_manifests \
--task audio_finetuning \
--nbest 1 \
--path /home/zlf/Semi-Supervised-Speech-Recognize/chechpint/wenetspeech_phoneme/checkpoint_last.pt \
--gen-subset dev \
--results-path /home/zlf/Semi-Supervised-Speech-Recognize/result \
--w2l-decoder viterbi  \
--lm-weight 2 \
--word-score -1 \
--sil-weight 0 \
--criterion ctc \
--labels phe \
--max-tokens 1000000 \
--post-process letter