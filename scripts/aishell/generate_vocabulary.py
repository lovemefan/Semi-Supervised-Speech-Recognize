# -*- coding: utf-8 -*-
# @Time  : 2021/10/21 9:48
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : prepare_data.py
import argparse


def generate_vocabulary(tsv_path: str, vocab_path: str):
    """
    generate vocabulary file from tsv file
    :param tsv_path:
    :param vocab_path:
    :return:
    """
    vocab = {}
    with open(tsv_path, 'r', encoding='utf-8') as tsv:
        for line in tsv:
            path, text = line.split('\t')
            for char in text:
                if char != ' ' and char != '\n':
                    if not vocab.get(char, None):
                        vocab[char] = 1
                    else:
                        vocab[char] += 1

    with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
        for item, value in sorted(vocab.items(), key=lambda x: x[1], reverse=True):
            vocab_file.write(f"{item}\t{value}\n")
    print(f"{vocab_path} created.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--tsv", required=True, help="input you tsv file")
    parser.add_argument("--vocab_output", required=True, help="input the path of vocab file output")
    args = parser.parse_args()

    generate_vocabulary(args.tsv, args.vocab_output)
