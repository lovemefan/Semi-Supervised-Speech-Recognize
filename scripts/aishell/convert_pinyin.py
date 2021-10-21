# -*- coding: utf-8 -*-
# @Time  : 2021/10/21 15:44
# @Author : lovemefan
# @Email : lovemefan@outlook.com
# @File : convert_pinyin.py
import argparse
import json
import struct

from pypinyin import lazy_pinyin, Style
from tqdm import tqdm


def get_audio_info(file_path):
    """
    get wav format file information more faster
    :param file_path:
    :return:
    """
    with open(file_path, 'rb') as file:
        riff_chunk_id = file.read(4)
        if riff_chunk_id != b'RIFF':
            raise Exception("No RIFF header found")
        riff_chunk_size = struct.unpack('<I', file.read(4))[0]

        wave_format = file.read(4)
        if wave_format != b"WAVE":
            raise Exception("Not a WAVE file")

        # read fmt chunk
        fmt_chunk_id = file.read(4)
        if fmt_chunk_id != b'fmt ':
            raise Exception("fmt chunk missing")

        fmt_chunk_size = struct.unpack('<I', file.read(4))[0]
        audioformat = struct.unpack('<H', file.read(2))[0]
        numchannels = struct.unpack('<H', file.read(2))[0]
        samplerate = struct.unpack('<I', file.read(4))[0]
        byterate = struct.unpack('<I', file.read(4))[0]
        bytespersample = struct.unpack('<H', file.read(2))[0]
        bitspersample = struct.unpack('<H', file.read(2))[0]
        if fmt_chunk_size == 18:
            extraparams = struct.unpack('<H', file.read(2))[0]
            fact_chunk_id = file.read(4)
            if fact_chunk_id != b'fact':
                raise Exception("fact chunk missing")

            fact_chunk_size = struct.unpack('<I', file.read(4))[0]
            samplelength = struct.unpack('<I', file.read(4))[0]
        data_chunk_id = file.read(4)

        duration = (riff_chunk_size - 36) / (bytespersample * numchannels * samplerate)

        return bitspersample, samplerate, numchannels, duration


def convert_tsv_to_json(tsv_path: str, tsv_output: str):
    """convert tsv file that transform chinese into phoneme
    """
    datas = []
    with open(tsv_path, 'r', encoding='utf-8') as tsv_file:
        lines = [line for line in tsv_file]
        for line in tqdm(lines):
            path, text = line.split('\t')
            phoneme = ' '.join([i for i in lazy_pinyin(text, errors='ignore', style=Style.TONE3)])
            bits, sample_rate, channel, duration = get_audio_info(path)

            data = {
                'audio_filepath': path,
                'duration': duration,
                'text': text,
                'phoneme': phoneme,
            }
            datas.append(data)

    with open(tsv_output, 'w', encoding='utf-8') as tsv_output:
        json.dump(tsv_output, ensure_ascii=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv", required=True, help="input you tsv file")
    parser.add_argument("--tsv_output", required=True, help="input the dir path of  file output")
    args = parser.parse_args()
    convert_tsv_to_json(args.tsv,  args.tsv_output)