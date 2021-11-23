fairseq-hydra-train \
    +task.data=/root/dataset/speechDataset/wenetspeech-s/wav2vec_manifests \
    common.tensorboard_logdir=/root/data/Semi-Supervised-Speech-Recognize/tensorboard/wenetspeech \
    distributed_training.distributed_world_size=1 \
    optimization.update_freq=[24] \
    checkpoint.save_dir=/root/data/Semi-Supervised-Speech-Recognize/chechpint/wenetspeech \
    model.w2v_path=/root/dataset/speechDataset/wenetspeech-s/wenetspeech_wav2vec2_model/wenetspeech_wav2vec2_model/checkpoint_best.pt \
    dataset.num_workers=2 \
    dataset.max_tokens=1200000 \
    --config-dir   /root/dataset/speechDataset/wenetspeech-s/wenetspeech_wav2vec2_model/wenetspeech_wav2vec2_model \
    --config-name  wenet_test
