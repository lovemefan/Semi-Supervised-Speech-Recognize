# -*- coding: utf-8 -*-
# @Time  : 2021/11/5 13:49
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : unispeech_manifest.py
import json
import os


def generate_phoneme_vocabulary(token_path, output_file):
    """
    :param toekn_path:
    :param output_file:
    :return:
    """
    tokens = {}
    count = 0
    with open(token_path, 'r', encoding='utf-8') as file:
        with open(output_file, 'w', encoding='utf-8') as output:
            for line in file:
                line = line.strip()
                if line not in tokens.keys():
                    tokens[line] = count
                    count += 1

            json.dump(tokens, output, ensure_ascii=False)

    with open(os.path.join(os.path.dirname(output_file), 'dict.phe.txt'), 'w', encoding='utf-8') as output:
        data = [f"{key} {value}" for key, value in tokens.items()]
        output.write('\n'.join(data))



if __name__ == '__main__':
    generate_phoneme_vocabulary('/home/dataset/aishell/data_aishell/transcript/tokens_all.txt',
                                '/home/dataset/aishell/data_aishell/transcript/phoneme_vocab.json')