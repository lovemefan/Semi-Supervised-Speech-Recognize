# -*- coding: utf-8 -*-
# @Time  : 2021/11/8 22:01
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : wenet_vocab.py
import argparse

from tqdm import tqdm


def generate_vocab(file, output_file):
    lines = []
    with open(file, 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.strip())

    vocab = {}

    print(f"read to count frequency ...")
    for line in tqdm(lines):
        keys = line.split(' ')
        for key in keys:
            if vocab.get(key, None) is None:
                vocab[key] = 1
            else:
                vocab[key] += 1
    vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    print(f"write file {output_file}")
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write('\n'.join([f"{value[0]} {value[1]}"for value in vocab]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--label", required=True, help="input you scp file")
    parser.add_argument("--output", required=True, help="input the path of dir file output")
    args = parser.parse_args()
    generate_vocab(args.label, args.output)
