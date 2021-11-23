wenetspeech_root='/root/dataset/speechDataset/wenetspeech-s/data_s'
train_scp=$wenetspeech_root/train_s/wav.scp
train_text=$wenetspeech_root/train_s/text
test_scp=$wenetspeech_root/test_meeting/wav.scp
test_text=$wenetspeech_root/test_meeting/text
dev_scp=$wenetspeech_root/test_net/wav.scp
dev_text=$wenetspeech_root/test_net/text
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