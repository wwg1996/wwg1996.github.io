---
layout: post
title: How to Make a Text Summarizer
date: 2018-03-07 13:34:59 +0800
categories: AI
tags: Siraj NLP 
img: http://wangweiguang.xyz/images/ML.jpg
---

[TOC]

课程视频见:[https://www.bilibili.com/video/av20427016/](https://www.bilibili.com/video/av20427016/)

## 总结
* **问题**：将输入一篇文章自动生成标题。
* 背景：自动文摘，即如何把大量的文本内容总结为一段或几段话。这是自然语言理解的一个重要应用方面，无论是将大量的文字进行信息抽取，或者是非文本的数据总结为人类可读的形式，都可以通过机器学习的方法来实现。（应用领域：气象，金融，医疗等）
* **语料**：一万篇英文新闻文章及其标题。
* **技术**：Word2vec，GloVe
* **模型**：seq2seq，LSTM

## 词向量技术
将单词转化为向量形式，不同的单词对应不同的向量，创建词向量让我们可以通过数学方法处理词汇。
![image](http://wangweiguang.xyz/images/w2v.jpg)

**举例**：四个单词及其词向量，每个词向量两个维度，即每个词用两个因素描述，这里可以理解为（性别，地位）。这样我们可以把这些向量代表词汇在坐标轴上描绘出来，可以从距离看出两个单词的相似性，甚至可以对这些单词进行数学运算比如说：国王+男人-女人=？

![image](http://wangweiguang.xyz/images/w2v1.jpg)
![image](http://wangweiguang.xyz/images/w2v2.jpg)

## word2vec
word2vec是一种单词向量化的技术；Word2vec是一个通过大量标签文本型语料进行训练的两层神经网络模型，
也是一个你可以下载的预训练模型。

## GloVe
另一种词向量技术是基于计数的，GloVe算法便是其中的一个代表。它首先通过上下文构造一个词的大的共发生矩阵，每个格的数值代表了两个单词在一些内容中出现的频率（此处为句子中）。这样细想了就可以表示为每行（或列）的数字组合了。

![image](http://wangweiguang.xyz/images/gfsjz.jpg)

## seq2seq
seq2seq是一种神经网络模型，他的输出和输入都可以为一个序列（长度可以变化），比如，再这里输入是文章，输出是标题。主要的组成有编码器和解码器，如下图。

![image](http://wangweiguang.xyz/images/s2s.jpg)

## LSTM与注意力机制
在编码器与解码器中间还有一个模块提供注意力机制，这是由LSTM单元实现的。学习理论的一个重要方面是注意力，即我们要记住的最重要，最相关的单词（数据）是那些？注意力模块计算了每个输入词的权重，以决定该输入应该注意多少。

![image](http://wangweiguang.xyz/images/attention.jpg)

## 编程实现
在vocabulary-embedding里下载glove.6B的时候,出现错误提示“'unzip' 不是内部或外部命令，也不是可运行的程序或批处理文件。”，前面有一回是成功了的，但是下载的文件太大中途卡住了。

有缘再回来解决吧。