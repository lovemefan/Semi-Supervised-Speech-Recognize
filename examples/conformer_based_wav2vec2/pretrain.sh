fairseq-hydra-train \
  +task.data=/root/dataset/speechDataset/wenetspeech-s/wav2vec_manifests \
  common.tensorboard_logdir=/root/data/Semi-Supervised-Speech-Recognize/tensorboard/wenetspeech/pretrain \
  distributed_training.distributed_world_size=1 \
  checkpoint.save_dir=/root/data/Semi-Supervised-Speech-Recognize/chechpint/wenetspeech/pretrain \
  model._name=conformer_based_wav2vec2 \
  --config-dir   /root/data/Semi-Supervised-Speech-Recognize/examples/conformer_based_wav2vec2/config/pretraining \
  --config-name   conformer_based_wav2vec2