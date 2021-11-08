train_scp='/root/dataset/speechDataset/wenetspeech-s/data_s/train_s/wav.scp'
train_text='/root/dataset/speechDataset/wenetspeech-s/data_s/train_s/text'
test_scp='/root/dataset/speechDataset/wenetspeech-s/data_s/test_meeting/wav.scp'
test_text='/root/dataset/speechDataset/wenetspeech-s/data_s/test_meeting/text'
dev_scp='/root/dataset/speechDataset/wenetspeech-s/data_s/test_net/wav.scp'
dev_text='/root/dataset/speechDataset/wenetspeech-s/data_s/test_net/text'
output_dir='/root/dataset/speechDataset/wenetspeech-s/wav2vec_manifests'
labels='phe'
# generate train subset manifest
python utils/wenet_manifest.py --scp $train_scp --text $train_text --subset train --output_dir $output_dir
# generate test subset manifest
python utils/wenet_manifest.py --scp $test_scp --text $test_text --subset test --output_dir $output_dir
# generate dev subset manifest
python utils/wenet_manifest.py --scp $dev_scp --text $dev_text --subset dev --output_dir $output_dir
# generate vocab
python utils/wenet_vocab.py --label $output_dir/train.$labels --output $output_dir/dict.$labels.txt