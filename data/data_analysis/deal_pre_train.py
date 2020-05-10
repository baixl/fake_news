# coding:utf-8
'''
处理预训练的词向量
'''
import re
import random
import collections
from tqdm import tqdm
import numpy as np
import pickle as pkl

words = set()
word_to_vec = {}
with open("../task1/sgns.sogou.char", 'r', encoding="utf-8") as f:
    for line in f:
        line = line.strip().split()
        curr_word = line[0]
        if len(curr_word) >= 2:
            continue  # 只处理单字
        words.add(curr_word)
        word_to_vec[curr_word] = np.array(line[1:], dtype=np.float32)
    print(len(words))

vocab = pkl.load(open("../task1/vocab.pkl", "rb"))  # 返回dict,  word：id

VOCAB_SIZE = len(vocab)
print(VOCAB_SIZE)
EMBEDDING_SIZE = 300
static_embeddings = np.zeros([VOCAB_SIZE, EMBEDDING_SIZE])

for word, id in vocab.items():
    # 用词向量填充，如果没有对应词向量，就随机填充
    word_vector = word_to_vec.get(
        word, 4 * np.random.random(EMBEDDING_SIZE) - 2)
    static_embeddings[id, :] = word_vector
# 填充pad
pad_id = vocab['<pad>']
static_embeddings[pad_id, :] = np.zeros(EMBEDDING_SIZE)

static_embeddings = static_embeddings.astype(np.float32)
np.save("../task1/sogou_embedding.npy", static_embeddings)
