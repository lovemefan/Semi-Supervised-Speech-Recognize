# -*- coding: utf-8 -*-
# @Time  : 2021/10/21 16:08
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : convert_audio.py
import argparse
import os

from tqdm import tqdm


def convert_into_pcm_16bit_mono(input_file, output_file):
    """
    use ffmpeg to convert audio into 16 bit mono pcm format audio
    :param input_file:
    :param output_file:
    :return:
    """
    cmd = f"ffmpeg -i {input_file} -acodec pcm_s16le -ac 1 -ar 16000 {output_file} -y"
    os.system(cmd)


def convert(tsv_path: str, audio_output_dir: str, tsv_ouput_dir: str):
    """
    input tsv file to convert all audio
    :param tsv_ouput_dir:
    :param audio_output_dir:
    :param tsv_path:
    :return:
    """
    target_tsv_path = os.path.join(tsv_ouput_dir, f"{os.path.basename(tsv_path).replace('.tsv', '')}.json")
    with open(tsv_path, 'r', encoding='utf-8') as tsv_file, \
            open(target_tsv_path, 'w', encoding='utf-8') as target_tsv:

        for line in tqdm(tsv_file):
            try:
                path, text = line.split('\t')
            except ValueError:
                break
            target_audio_path = os.path.join(audio_output_dir, os.path.basename(path))
            target_tsv.write(f"{target_audio_path}\t{text}")
            if not os.path.exists(target_audio_path):
                convert_into_pcm_16bit_mono(path, target_audio_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, help="input you tsv file")
    parser.add_argument("--audio_output", required=True, help="input the dir path of  file output")
    parser.add_argument("--tsv_output", required=True, help="input the dir path of  file output")
    args = parser.parse_args()
    convert(args.tsv, args.audio_output, args.tsv_output)
