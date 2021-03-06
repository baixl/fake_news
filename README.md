
### 赛题地址
https://www.biendata.com/competition/falsenews/data/

### 数据集
```
虚假新闻检测：
1、训练集共包含38,471条新闻，其中包含真实新闻19,186条，虚假新闻19,285条。==> 数据比较均衡
初赛测试集共4,000条，复赛测试集3,902条，真假新闻比例与训练集基本一致。

数据字段：
id：新闻id，每条文本中id均不相同，唯一表征一条新闻；
text: 新闻的文本内容；（部分内容不规整）
label: 取值为{0,1}，0表示真实新闻，1表示虚假新闻。

原始数据集合：
 train.csv 
 test_stag1.csv
```

### 数据处理
```
 1、
 数据规整后的训练集： 
 train_new.csv
 test_stag1_new.csv

2、数据分析
平均中文长度约100，通过画出文本长度的分布图，可以看出大概取长度180左右截断，比较合适

3、数据预处理
数据中，标点符号、常用停用词可以去除

停用词表：https://github.com/baixl/stopwords， 注意， github上的停用词需要筛选下，比如这里的正样本中， 可能就是废话单词"一些、有些"这种较多，那就不能过停用词
可以只过滤单字且无意义的词

但正样本(fake新闻)中，往往都包好表情符号，需要保留(但中文词向量中不一定有)

4、样本集合只有3800条， 样本太小，embeeding需要额外的词向量。
    先用word2vec尝试，后续使用字级别的bert/ERINE(百度)
    中文词向量地址：https://github.com/baixl/Chinese-Word-Vectors
    下载 搜狗新闻  Sogou News 搜狗新闻		300d（word+char）

```

### 评测方法
```
评测方法
F值：

对虚假新闻文本检测任务和虚假新闻多模态检测任务，评估指标为测试集中虚假新闻类别上的F1值，是正确率和召回率的调和平均值：

正确率 = 预测出的真正虚假新闻样本数 / 所有被预测为虚假新闻的样本数
召回率 = 预测出的真正虚假新闻样本数 / 所有真正的虚假新闻样本数
F1值 = （2 * 正确率 * 召回率）/（正确率 + 召回率）

测试集分为初赛测试集和复赛测试集两部分。选手在初赛过程中可以看到初赛测试集的评测成绩，但确定最终成绩的是复赛测试集的分数。复赛阶段，主办方将根据复赛参赛队伍提交的模型和代码，检验结果的可复现性，并根据复现结果确定复赛测试集的分数。
对于分数相同的情况，根据初赛和复赛总提交次数确定名次，总提交次数较少的队伍排名靠前。
```