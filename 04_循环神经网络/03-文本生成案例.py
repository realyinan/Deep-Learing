import torch
import re
import jieba
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


def build_vocab():
    # 数据集位置
    file_name = './data/jaychou_lyrics.txt'
    # 唯一词列表
    unique_words = []
    # 每行文本分词列表
    all_words = []
    for line in open(file=file_name, mode='r', encoding='utf-8'):
        words = jieba.lcut(line)
        all_words.append(words)
        for word in words:
            if word not in unique_words:
                unique_words.append(word)
    word_count = len(unique_words)
    # 词到索引映射
    word_to_index = {word: idx for idx, word in enumerate(unique_words)}
    # 歌词文本用词表索引表示
    corpus_idx = []
    for words in all_words:
        # 临时储存每行词的索引下标
        temp = []
        for word in words:
            temp.append(word_to_index[word])
        temp.append(word_to_index[' '])  # 将每行\n和下一行分开
        corpus_idx.extend(temp)
    return unique_words, word_to_index, word_count, corpus_idx



if __name__ == "__main__":
    unique_words, word_to_index, word_count, corpus_idx = build_vocab()
    print(corpus_idx)
    print(unique_words)
    print(word_to_index)
    print(word_count)