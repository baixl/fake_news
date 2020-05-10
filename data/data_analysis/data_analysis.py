#coding:utf-8
'''
数据分析
'''
import matplotlib.pyplot as plt
import numpy as np

def read_data(train_path):
    with open(train_path, 'r') as f:
        cnt = 0
        idx = []
        idx, text, label = [], [], []
        for line in f:
            cnt +=1
            if cnt ==1 :continue # 标签行
            line_arr = line.split("\t")
            if len(line_arr) == 3:
                idx.append(line_arr[0])
                text.append(line_arr[1])
                label.append(line_arr[2])
            else:
                raise "train data length error!!!"
        return idx, text, label
          

def  get_len(train_text_array):
    return [len(item) for item in train_text_array]

def plot(train_len, png_name):
    save_path = './' + png_name

    plt.figure()
    plt.title(png_name)
    plt.xlabel('train_txt_length')
    plt.ylabel('Count')
    plt.hist(train_len, bins=50, range=[1,500], alpha=0.3, color='r')
    # plt.legend()
    # plt.show()
    plt.savefig(save_path)



if __name__ == "__main__":
    train_path = "../task1/train_new.csv"
    idx, text, label = read_data(train_path)
    text_len = np.array(get_len(text))
    print(np.std(text_len)) # 99.6330281646925 , 未做任何处理，长度均值99
    plot(get_len(text), 'tain_len_distribution.png')

