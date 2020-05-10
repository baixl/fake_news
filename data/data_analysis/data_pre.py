# coding:utf-8
'''
数据预处理
1、部分数据不规则，cont有多行，这里打平成一行
2、cont部分包含 引文的逗号, 这里替换成空格
3、将逗号分隔转换成\t分隔
4、输出文件 train_news.csv
'''
import re
import random
import collections
from tqdm import tqdm


def remove_punctuation(line):
    '''
    去除所有半角全角符号，只留字母、数字、中文。
    '''
    rule = re.compile("[^a-zA-Z0-9\u4e00-\u9fa5]")
    line = rule.sub('', line)
    return line


def process_train(train_path, train_out_path):
    file = open(train_out_path, 'w', encoding='utf-8')
    # with open(train_path, 'r') as f:
    f = open(train_path, 'r', encoding='utf-8').readlines()
    cur_line = ''
    for line in f:
        line = line.replace("\n", '').replace(
            "\r", "").replace("\r\n", "").replace("\t", "")

        if line.endswith(',0') or line.endswith(',1') or line.endswith(',label'):
            cur_line += line
            line_arr = cur_line.split(',')
            id = line_arr[0]
            label = line_arr[-1]

            text = ' '.join(line_arr[1:-1])
            text = remove_punctuation(text)
            file.write("\t".join([str(label), text]) + '\n')
            cur_line = ''
        else:
            cur_line += line
    file.close()


def process_test(test_path, test_out_path):
    file = open(test_out_path, 'w', encoding='utf-8')
    # with open(test_path, 'r') as f:

    for line in open(test_path, 'r', encoding='utf-8').readlines():
        line = line.replace("\n", '').replace(
            "\r", "").replace("\r\n", "").replace("\t", "")
        line_arr = line.split(",")
        id = line_arr[0]
        text = ' '.join(line_arr[1:])
        text = remove_punctuation(text)
        file.write("\t".join([str(id), text]) + '\n')
    file.close()


def split_train_dev(train_out_path):
    '''
    将预处理后的数据，分割成 train、 dev、 test 3份，总共数据38471条 按照8:1:1, 约 30000:4000:4000切分
    '''
    file = open(train_out_path, 'r', encoding='utf-8').readlines()
    data = []
    for line in file:
        if line.startswith('label'):
            continue
        data.append(line)
    random.shuffle(data)
    train_data = data[0:30000]
    dev_data = data[30001:34000]
    test_data = data[34000:-1]
    write_data(train_data, "../task1/train_data.csv")
    write_data(dev_data, "../task1/dev_data.csv")
    write_data(test_data, "../task1/test_data.csv")


def write_data(data, path):
    file = open(path, 'w', encoding='utf-8')
    for line in data:
        file.write(line)
    file.close()


if __name__ == "__main__":
    train_path = "../task1/train.csv"
    train_out_path = "../task1/train_new.csv"
    test_path = "../task1/test_stage1.csv"
    test_out_path = "../task1/test_stage1_new.csv"
    process_train(train_path, train_out_path)
    process_test(test_path, test_out_path)

    split_train_dev(train_out_path)
