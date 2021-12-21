fairseq-hydra-train \
    +task.data=/root/dataset/speechDataset/wenetspeech-s/wav2vec_manifests \
    common.tensorboard_logdir=/root/data/Semi-Supervised-Speech-Recognize/tensorboard/wenetspeech/finetuning \
    distributed_training.distributed_world_size=1 \
    optimization.update_freq=[24] \
    checkpoint.save_dir=/root/data/Semi-Supervised-Speech-Recognize/chechpint/wenetspeech/finetuning \
    model.w2v_path=/root/data/Semi-Supervised-Speech-Recognize/chechpint/wenetspeech/checkpoint_best.pt \
    dataset.num_workers=2 \
    dataset.max_tokens=1200000 \
    --config-dir   /root/data/Semi-Supervised-Speech-Recognize/examples/conformer_based_wav2vec2/config/finetuning \
    --config-name  conformer_based_wav2vec2
