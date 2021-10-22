fairseq-hydra-train \
    task.data=/root/dataset/speechDataset/aishell-1/data_aishell/transcript \
    common.tensorboard_logdir=/root/data/Semi-Supervised-Speech-Recognize/tensorboard  \
    distributed_training.distributed_world_size=1 \
    +optimization.update_freq='[24]' \
    model.w2v_path=/root/dataset/speechDataset/pretrain-model/wav2vec2_large_100k.pt \
    dataset.num_workers=2 \
    dataset.max_tokens=2800000 \
    --config-dir /root/data/Semi-Supervised-Speech-Recognize/examples/wav2vec/config/finetuning \
    --config-name aishell