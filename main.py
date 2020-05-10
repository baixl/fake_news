# coding:utf-8
import torch
import time
import numpy as np
import argparse
from importlib import import_module
from data_load import build_dataset, build_iterator
from data_load import get_time_dif
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from sklearn import metrics

parser = argparse.ArgumentParser(description="fake news identify")
parser.add_argument("--model",
                    type=str,
                    required=True,
                    help=" TextCNN TextRNN FastText TextRCNN Transformer bert")
args = parser.parse_args()


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    total_batch = 0
    last_imporve = 0
    early_stop_flag = False
    dev_best_loss = float("inf")
    best_acc_val = 0.0  # 最佳验证集准确率

    writer = SummaryWriter(log_dir=config.log_path + '/' +
                           time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print("epoch: {} / {}".format(epoch+1, config.num_epochs))

        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            if total_batch % 100 == 0:
                label_data = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(label_data, predict)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)

                if best_acc_val < dev_acc:
                    dev_best_loss = dev_loss
                    best_acc_val =  dev_acc
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc,
                                 dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)

                model.train()

            total_batch += 1
            # 判断早停
            if total_batch - last_imporve > config.require_improvement:
                print("early stop for epoch: {}".format(
                    config.require_improvement))
                early_stop_flag = True
                break
        if early_stop_flag:
            break
    writer.close()
    test(config, model, test_iter)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    # if test:
    #     report = metrics.classification_report(
    #         labels_all, predict_all, target_names=config.class_list, digits=4)
    #     confusion = metrics.confusion_matrix(labels_all, predict_all)
    #     return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)

def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss = evaluate(config, model, test_iter)
    msg = "\n Test loss:{0:5.2} Test Acc:{1:6.2%}"
    print(msg.format(test_loss, test_acc))


if __name__ == "__main__":
    data_set = "data/task1"
    embedding = "sgns.sogou.char"
    model_name = args.model
    model = import_module("models." + model_name)  # 导入model
    config = model.Config(embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    #  加载数据
    start_time = time.time()
    print("1、加载数据...\n")

    vocab, train_data, dev_data, test_data = build_dataset(config)
    config.n_vocab = len(vocab)
    print('vocab size :', config.n_vocab)

    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    time_dif = get_time_dif(start_time)
    print("\n加载数据耗时: ", time_dif)

    # 开始训练
    print("2、开始训练...\n")
    start_time = time.time()
    model = model.Model(config).to(config.device)
    print("------\nmodel : {0}\n".format(model_name))
    print(model.parameters)
    print("------\ntraining...\n")

    train(config, model, train_iter, dev_iter, test_iter)

    time_dif = get_time_dif(start_time)
    print("\n训练耗时: ", time_dif)
