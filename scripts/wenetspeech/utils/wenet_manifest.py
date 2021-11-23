# -*- coding: utf-8 -*-
# @Time  : 2021/11/5 13:49
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : unispeech_manifest.py
import argparse
import json
import os
import jieba
from pypinyin import pinyin, lazy_pinyin, Style
from tqdm import tqdm
import soundfile as sf

def generate_manifests(scp_file, text_file, output_file, subset='train'):
    """
    :param toekn_path:
    :param output_file:
    :return:
    """
    files = {}
    print(f"reading {scp_file} file...")
    lines = []
    with open(scp_file, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line)
    print(f"reading {scp_file} file done")

    print(f"read file name and path")
    for line in tqdm(lines):
        name, path = line.split(' ')
        files[name] = {'path': path.strip().replace('/home/speech/Audio/WenetSpeech/wenetspeech/audio_seg_s', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(scp_file))), 'audio_seg_s'))}

    print(f"reading {text_file} file...")
    lines.clear()
    with open(text_file, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line)
    print(f"reading {text_file} file done")

    jieba.cut('jieba test', use_paddle=True)
    print(f"read transcription")
    for line in tqdm(lines):
        name, text = line.strip().split(' ', 1)
        files[name]['sentence'] = text
        seg_list = jieba.cut(text, use_paddle=True)
        cutted = '|'.join([i for i in seg_list if i != '' and i != ' '])
        files[name]['word'] = cutted + '|'
        files[name]['letter'] = ' | '.join([' '.join(list(item)) for item in cutted.split('|')]) + " |"
        files[name]['phoneme'] = ' | '.join(convert_into_phoneme(item) for item in cutted.split('|')) + " |"

    file_name = os.path.basename(scp_file).replace('.scp', '')
    file_path = os.path.join(output_file, f'{subset}.tsv')
    print(f"writing {file_path} ...")
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as tsv:
            lines.clear()
            tsv.write(f"path\tnframes\n")
            for key, value in tqdm(files.items()):
                nframes = sf.info(value['path']).frames
                lines.append(f"{value['path']}\t{nframes}")
            tsv.write('\n'.join(lines))

    file_path = os.path.join(output_file, f'{subset}.wrd')
    print(f"writing {file_path} ...")
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as wrd:
            lines.clear()
            for key, value in tqdm(files.items()):
                lines.append(f"{value['word']}")
            wrd.write('\n'.join(lines))

    file_path = os.path.join(output_file, f'{subset}.ltr')
    print(f"writing {file_path} ...")
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as ltr:
            lines.clear()
            for key, value in tqdm(files.items()):
                lines.append(f"{value['letter']}")
            ltr.write('\n'.join(lines))

    file_path = os.path.join(output_file, f'{subset}.phe')
    print(f"writing {file_path} ...")
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as phe:
            lines.clear()
            for key, value in tqdm(files.items()):
                lines.append(f"{value['phoneme']}")
            phe.write('\n'.join(lines))


def convert_into_phoneme(text):
    initals = pinyin(text, style=Style.INITIALS)
    finals = pinyin(text, style=Style.FINALS_TONE3)
    first_letter = pinyin(text, style=Style.FIRST_LETTER)
    initals = [initals[i] if initals[i][0] != '' else first_letter[i] for i in range(len(finals))]
    assert len(initals) == len(finals)
    phoneme = [f"{initals[i][0]} {finals[i][0]}" for i in range(len(initals))]
    return ' '.join(phoneme)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--scp", required=True, help="input you scp file")
    parser.add_argument("--text", required=True, help="input the path of text output")
    parser.add_argument("--output_dir", required=True, help="input the path of dir file output")
    parser.add_argument("--subset",  default="test", help="train, dev, test")
    args = parser.parse_args()

    generate_manifests(args.scp, args.text, args.output_dir, args.subset)
