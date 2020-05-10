# coding: UTF-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FastText(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        super(Config, self).__init__(dataset, embedding)
        self.model_name = 'FastText'

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度： 小于该值则填充，大于则截断
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度

        self.hidden_size = 256
        self.n_gram_vocab = 250499


class Model(nn.Module):
    def __init__(self, config):
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(
                config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding_ngram_2 = nn.Embedding(
            config.n_gram_vocab, config.embed)
        self.embedding_ngram_3 = nn.Embedding(
            config.n_gram_vocab, config.embed)

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed * 3, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram_2(x[2])
        out_trigram = self.embedding_ngram_2(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out