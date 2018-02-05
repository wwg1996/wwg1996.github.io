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

