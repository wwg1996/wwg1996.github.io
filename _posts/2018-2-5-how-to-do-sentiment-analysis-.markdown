---
layout: post
title: 用机器学习实现情感分析
date: 2018-02-05 20:13:29 +0800
categories: AI
tags: 深度学习 Siraj 
img: http://wangweiguang.xyz/images/ML.jpg
---

* 
{:toc}

教程链接：[https://www.bilibili.com/video/av19178430/?spm_id_from=333.23.home_video_list.1](https://www.bilibili.com/video/av19178430/?spm_id_from=333.23.home_video_list.1)

## 知识
1. 情感分析两种方法：
   * 基于词典的方法：先对句子进行分词，然后统计个个词汇的个数，最后在情感字典中查找这些单词对应的情感值，然后可以计算出总体的情感。
   * **机器学习的方法**：输入大量句子以及这些句子的情感标签，就可以训练一个句子情感分类器，预测新的句子的情感。
   * 机器学习方法的优点：机器学习对情感分析会更为精准，深度神经网络可以很好的分辨出一些反讽语气的句子，这些句子的情感不是通过简单的表面词汇分析可以理解的。
2. 前馈过程接受固定大小的输入，比如二进制数；递归网络可以接受序列数据，比如文本。
3. 使用AWS（亚马逊云服务）让你的代码在云端上更快更方便的运行。

## 实例
这里的实例用到的库是tflearn，tflearn是一个深度学习库，他基于TensorFlow，并且提供了更高级的API，可以很好的帮助初学者入门深度学习。
```python
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb        # Internet Movie Database
```
### 数据导入
* pkl：字节流形式数据，更容易转换为其他python对象
* 取10000单词，10%的的数据作为验证集
```python
train, test, _ = imdb.load_data(path='imdb.pkl',  
                                n_words=10000,     
                                valid_portion=0.1) 
```

* 将数据划分为评论集和标签集
```python
trainX, trainY = train
testX, testY = test
```

### 数据处理
* 不能直接将文本数据中的字符串输入神经网络，必须先进行向量化，
* 神经网络作为一种算法，本质上还是对矩阵进行运算，
* 因此，将它们转换为数值或向量表示是必要的。
* pad_sequences的作用是把输入转换为矩阵的形式，并且对矩阵进行扩充。
* 矩阵的扩充是为了保持输入维数的一致性。
* 下面的参数标明了输入的数列会扩充到100的长度，扩充的部分数值为0。
```python
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
```

* 把评论集转位二进制向量（表示评价是积极或消极）
```python
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)
```

### 构造网络
1. 定义输入层，输入数据长度为100
2. 定义嵌入层，第一个参数是这一层接受的向量，即上一层输出的向量，共导入10000个单词，输出维度定义为128
3. 定义LSTM（Long short term memory）层，使我们的网络能够记住序列一开始的数据，将把dropout设置为0.08，这是一种防止过拟合的技术。
4. 定义全连接网络层，激活函数使用softmax。
5. 对于输入做回归操作，定义优化方法，与学习率，还有损失值计算方法
```python
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')
```

### 训练网络
1. 初始化神经网络
2. 训练神经网络，输入训练集与验证集，show_metric=True可以输出训练日志
```python
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)
```
因为注册AWS还要国外的信用卡，没有弄成，在自己笔记本上运行了40分钟，才训练好，迭代了7040次，准确度最后达到0.9475，损失值从一开始的0.5左右到了0.15。以后还是得想办法找一个免费的云服务器，跑一些小程序。

## Challenge

The challenge for this video is to train a model on this dataset of video game reviews from IGN.com. Then, given some new video game title it should be able to classify it. You can use pandas to parse this dataset. Right now each review has a label that's either Amazing, Great, Good, Mediocre, painful, or awful. These are the emotions. Using the existing labels is extra credit. The baseline is that you can just convert the labels so that there are only 2 emotions (positive or negative). Ideally you can use an RNN via TFLearn like the one in this example, but I'll accept other types of ML models as well. 

You'll learn how to parse data, select appropriate features, and use a neural net on an IRL pr


```python
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences, VocabularyProcessor
import pandas as pd
import numpy as np

# 数据导入
# 用pd做数据导入，这里训练集测试集随机的抽取会更好
dataframe = pd.read_csv('ign.csv').iloc[:, 1:3]

train = dataframe.iloc[:int(dataframe.shape[0]*0.9), :]
test = dataframe.iloc[int(dataframe.shape[0]*0.9):dataframe.shape[0], :]

trainX = train.title
trainY = train .score_phrase
testX = test.title
testY = test.score_phrase

# 数据处理
# 和实例不同的是这里的数据是纯文本的，处理前要转换成数据序列，用到了tflearn中的VocabularyProcessor相关方法；样本集分为11类
vocab_proc = VocabularyProcessor(15)
trainX = np.array(list(vocab_proc.fit_transform(trainX)))
testX = np.array(list(vocab_proc.fit_transform(testX)))

vocab_proc2 = VocabularyProcessor(1)
trainY = np.array(list(vocab_proc2.fit_transform(trainY))) - 1
trainY = to_categorical(trainY, nb_classes=11)
vocab_proc3 = VocabularyProcessor(1)
testY = np.array(list(vocab_proc3.fit_transform(testY))) - 1
testY = to_categorical(testY, nb_classes=11)

# 构建网络
# 现在并不清楚要按什么标准构建不同的网络，直接用的实例
net = tflearn.input_data([None, 15])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')
            
# 训练网络
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)
```

## Jovian's Winning Code:
Jovian的冠军代码写的很全面了，每一步的解释也很详细。这里同样把这个当做了一个分类问题，比较了三种不同的分类方式的区别，训练完成最后的深度学习网络可以达到0.5左右的正确率，输入一个游戏的名称，可以预测出这个游戏的评价等级。

### 依赖库
```python
import pandas as pd
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
```
### 数据导入
导入游戏评价数据库ign.csv
```python
original_ign = pd.read_csv('ign.csv')
```
查看数据库的形状
```python
print('original_ign.shape:', original_ign.shape)
```
output：
```
original_ign.shape: (18625, 11)
```
共有18625个游戏的数据，每个游戏有11项信息。其中只有游戏的评价信息（score_phrase）是我们需要关注的。下面统计游戏的各种评价。
```python
original_ign.score_phrase.value_counts()
```
output
```
Great          4773
Good           4741
Okay           2945
Mediocre       1959
Amazing        1804
Bad            1269
Awful           664
Painful         340
Unbearable       72
Masterpiece      55
Disaster          3
Name: score_phrase, dtype: int64
```
可以看出评价为Great和Good的最多，而Disaster的评价只有3个。

### 数据处理
#### 预处理
检查属否有null元素（缺失项）
```python
original_ign.isnull().sum()
```
output：
```
Unnamed: 0         0
score_phrase       0
title              0
url                0
platform           0
score              0
genre             36
editors_choice     0
release_year       0
release_month      0
release_day        0
dtype: int64
```
将缺失值填充为空字符串（这个例子其实无需做这两步，但要养成检查缺失值的好习惯）：
```python
original_ign.fillna(value='', inplace=True)
```
#### 数据划分
划分样本集和标签集：
```python
X = ign.text
y = ign.score_phrase
```

分类训练集和测试集：
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
```

#### 样本集处理
将样本集的字符串转变为数字序列。创建vocab，把X转化为X\_word\_ids。
```python
vect = CountVectorizer(ngram_range=(1,1), token_pattern=r'\b\w{1,}\b')

vect.fit(X_train)
vocab = vect.vocabulary_

def convert_X_to_X_word_ids(X):
    return X.apply( lambda x: [vocab[w] for w in [w.lower().strip() for w in x.split()] if w in vocab] )

X_train_word_ids = convert_X_to_X_word_ids(X_train)
X_test_word_ids  = convert_X_to_X_word_ids(X_test)
```
序列扩充
```python
从       
X_test_padded_seqs  = pad_sequences(X_test_word_ids , maxlen=20, value=0)
```

#### 标签集处理
```python
unique_y_labels = list(y_train.value_counts().index)
le = preprocessing.LabelEncoder()
le.fit(unique_y_labels)

y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), nb_classes=len(unique_y_labels))
y_test  = to_categorical(y_test.map(lambda x:  le.transform([x])[0]), nb_classes=len(unique_y_labels))
```

### 构造网络
构造网络和实例一样
```python
n_epoch = 100
size_of_each_vector = X_train_padded_seqs.shape[1]
vocab_size = len(vocab)
no_of_unique_y_labels = len(unique_y_labels)
```
```python
net = tflearn.input_data([None, size_of_each_vector]) # The first element is the "batch size" which we set to "None"
net = tflearn.embedding(net, input_dim=vocab_size, output_dim=128) # input_dim: vocabulary size
net = tflearn.lstm(net, 128, dropout=0.6) # Set the dropout to 0.6
net = tflearn.fully_connected(net, no_of_unique_y_labels, activation='softmax') # relu or softmax
net = tflearn.regression(net, 
                         optimizer='adam',  # adam or ada or adagrad # sgd
                         learning_rate=1e-4,
                         loss='categorical_crossentropy')
```

### 训练网络
初始化
```python
model = tflearn.DNN(net, tensorboard_verbose=0)
```

训练
```python
model.fit(X_train_padded_seqs, y_train, 
           validation_set=(X_test_padded_seqs, y_test), 
           n_epoch=n_epoch,
           show_metric=True, 
           batch_size=100)
```
训练结果：
```python
Training Step: 16799  | time: 25.949s
| Adam | epoch: 100 | loss: 0.00000 - acc: 0.0000 -- iter: 16700/16759
Training Step: 16800  | time: 27.099s
| Adam | epoch: 100 | loss: 0.00000 - acc: 0.0000 | val_loss: 1.83644 - val_acc: 0.4326 -- iter: 16759/16759
--
```
