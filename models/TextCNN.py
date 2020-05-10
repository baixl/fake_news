# coding:utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    '''
    TextCNN 配置参数
    '''

    def __init__(self, data_path):
        self.model_name = "TextCNN"
        self.train_path = "./data/task1/train_data.csv"
        self.dev_path = "./data/task1/dev_data.csv"
        self.test_path = "./data/task1/test_data.csv"
        self.vocab_path = "./data/task1/vocab.pkl"
        self.save_path = "./data/save/" + self.model_name + ".ckpt"
        self.log_path = "./data/log/" + self.model_name

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')  # cuda
        print("cuda: ", torch.cuda.is_available())

        #TODO:
        self.embedding_pretrained = torch.tensor(
            np.load("./data/task1/sogou_embedding.npy").astype("float32"))
        self.dropout = 0.5
        self.require_improvement = 1000  # 早停轮数
        self.num_classes = 2  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 200  # 句子长度短填充长截断
        self.learning_rate = 1e-3  # 学习率
        self.embed_size = 300  # 字向量维度
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        Vocab = config.n_vocab  # 词表大小
        Dim = config.embed_size  # 词向量维度
        Cla = config.num_classes  # 输出类别数
        Ci = 1  # channel 数
        kernal_nums = config.num_filters
        Ks = config.filter_sizes

        # self.embedding = nn.Embedding(Vocab, Dim, padding_idx=Vocab - 1)
        self.embedding = nn.Embedding.from_pretrained(
            config.embedding_pretrained, freeze=True)

        self.convs = nn.ModuleList([
            nn.Conv2d(Ci, kernal_nums, (k, Dim))
            for k in Ks
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(kernal_nums * len(Ks), Cla)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv)
                         for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
