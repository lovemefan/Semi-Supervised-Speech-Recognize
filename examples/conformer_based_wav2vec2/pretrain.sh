fairseq-hydra-train \
  +task.data=/home/dataset/wenet_s/wav2vec_manifests \
  common.tensorboard_logdir=/root/data/Semi-Supervised-Speech-Recognize/tensorboard/wenetspeech \
  distributed_training.distributed_world_size=1 \
  model._name=conformer_based_wav2vec2 \
  --config-dir   /root/data/Semi-Supervised-Speech-Recognize/examples/conformer_based_wav2vec2/config/pretraining \
  --config-name   conformer_based_wav2vec2