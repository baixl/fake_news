# coding: utf-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm  # 进度条工具
import time
from datetime import timedelta

MAX_VOCAB_SIZE = 10000  # 词表长度限制
# TODO: 这三个字符， embedding里有吗
UNK, PAD, CLS = '<unk>', '<pad>',  '<cls>' # 未知字，padding符号


def build_vocab(train_path, tokenizer, max_size, min_freq):
    """
        1、词典按频次排序， 取频次大于 min_freq的部分构造词典， 并添加 unk 和pad
    """
    vocab_dict = {}
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            line = line.strip()
            if not line:
                continue
            line_arr = line.split("\t")
            if len(line_arr) != 2:
                continue
            text = line_arr[1]
            for word in tokenizer(text):
                vocab_dict[word] = vocab_dict.get(word, 0) + 1
        vocab_list = sorted(
            [item for item in vocab_dict.items() if item[1] >= min_freq],
            key=lambda x: x[1],
            reverse=True)[:max_size]

        vocab_dict = {
            word_count[0]: idx
            for idx, word_count in enumerate(vocab_list)
        }
        count = 0
        for k,v in vocab_dict.items():
            count+=1
            if count == 5:break
            print(k, v)

        vocab_dict.update({UNK: len(vocab_dict), PAD: len(
            vocab_dict) + 1, CLS: len(vocab_dict) + 1})
    return vocab_dict


def build_dataset(config):
    '''
    构造数据集
    '''
    def tokenizer(x):
        return [y for y in x]  # char-level, 以字为单位 默认

    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        vocab = build_vocab(config.train_path,
                            tokenizer=tokenizer,
                            max_size=MAX_VOCAB_SIZE,
                            min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))

    print("Vocab size: {0} ".format(len(vocab)))

    def load_dataset(path, pad_size=200):
        contents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                line = line.strip()
                if not line or len(line.split("\t")) != 2:
                    continue
                label, text = line.split('\t')

                tokens = tokenizer(text)
                seq_len = len(tokens)

                if len(tokens) < pad_size:
                    tokens.extend([vocab.get(PAD)] *
                                  (pad_size - len(tokens)))
                else:
                    tokens = tokens[:pad_size]
                    seq_len = pad_size
                # word to id
                words_line = []
                for word in tokens:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                contents.append((words_line, int(label), seq_len))  # 3列, words_line.未term转换成id的序列

        return contents  # 3列， (words_line、label、seq_len)

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


#TODO: 优化
class DatasetIterater(object):
    def __init__(self, dataset, batch_size, device, model_name):
        self.batch_size = batch_size  # 批量大小
        self.dataset = dataset  # 数据集
        self.n_batches = len(dataset) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device
        self.model_name = model_name

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        return (x, seq_len), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            dataset = self.dataset[self.index *
                                   self.batch_size:len(self.dataset)]
            self.index += 1
            dataset = self._to_tensor(dataset)
            return dataset

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            dataset = self.dataset[self.index *
                                   self.batch_size:(self.index + 1) *
                                   self.batch_size]
            self.index += 1
            dataset = self._to_tensor(dataset)
            return dataset

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size,
                           config.device, config.model_name)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
