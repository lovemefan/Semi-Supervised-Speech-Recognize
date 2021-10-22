# -*- coding: utf-8 -*-
# @Time  : 2021/10/21 15:44
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : convert_pinyin.py
import argparse
import json
import os
import struct

from pypinyin import lazy_pinyin, Style
from tqdm import tqdm
import soundfile as sf


def get_audio_info(file_path):
    """
    get wav format file information more faster
    :param file_path:
    :return:
    """
    audio_info = sf.info(file_path)
    return audio_info


def convert_txt_to_manifests(txt_path: str, output_dir: str):
    """convert tsv file that transform chinese into phoneme
    """
    datas = []
    with open(txt_path, 'r', encoding='utf-8') as txt_file:
        lines = [line.strip() for line in txt_file]

        # sentence = [d.split('\t')[1] for d in lines]
        #
        # words = [d.replace(' ', '|') for d in sentence]
        # letters = [' '.join(list(d)) + ' |' for d in words]
        # phonemes = [' '.join([i for i in lazy_pinyin(p, errors='ignore', style=Style.TONE3)]) for p in letters]

        tsv_output = os.path.join(output_dir, f"{os.path.basename(txt_path).replace('.txt', '')}.tsv")
        wrd_output = os.path.join(output_dir, f"{os.path.basename(txt_path).replace('.txt', '')}.wrd")
        ltr_output = os.path.join(output_dir, f"{os.path.basename(txt_path).replace('.txt', '')}.ltr")
        phe_output = os.path.join(output_dir, f"{os.path.basename(txt_path).replace('.txt', '')}.phe")

        with open(tsv_output, 'w', encoding='utf-8') as tsv_output_file, \
                open(wrd_output, 'w', encoding='utf-8') as wrd_output_file, \
                open(ltr_output, 'w', encoding='utf-8') as ltr_output_file, \
                open(phe_output, 'w', encoding='utf-8') as phe_output_file:
            print("creating manifests...")
            for line in tqdm(lines):
                path, sentence = line.split('\t')
                word = '|'.join(sentence.split(' '))
                wrd_output_file.write(word + '\n')
                letter = ' '.join(list(word))
                ltr_output_file.write(letter + '\n')
                phoneme = ' '.join([i for i in lazy_pinyin(letter, errors='ignore', style=Style.TONE3)])
                phe_output_file.write(phoneme + '\n')
                audio_info = get_audio_info(path)
                tsv_output_file.write(f"{path}\t{audio_info.frames}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", required=True, help="input you tsv file")
    parser.add_argument("--output_dir", required=True, help="input the dir path of  file output")
    args = parser.parse_args()
    convert_txt_to_manifests(args.txt, args.output_dir)
