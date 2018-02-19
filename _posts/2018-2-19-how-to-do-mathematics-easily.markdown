---
layout: post
title: How to Do Mathematics Easily
date: 2018-02-18 19:48:29 +0800
categories: AI
tags: 神经网络 Siraj 
img: http://wangweiguang.xyz/images/ML.jpg
---



课程链接：[https://www.bilibili.com/video/av19766394/](https://www.bilibili.com/video/av19766394/)

## 总结

1. 深度学习用到了线性代数，统计学和微积分。
2. 神经网络对输入的张量（tensor）进行一系列的运算来进行预测。
3. 我们可以用梯度下降法反向传播误差，相应的调制权重，从而优化预测。

4. 数据准备：收集和清洗数据的过程不需要数学，但是标准化的过程需要用到统计学知识。把各种输入（图片、文字、声音）转化成神经网络可以接受的张量形式这是线性代数的知识。
5. 构建网络：随机化初始参数的过程需要用到统计学知识。
6. 训练网络：前向传播过程用到矩阵乘法，误差反馈（梯度下降）的过程用到导数知识。

![image](http://wangweiguang.xyz/images/me3.jpg)

## 神经网络简例

```python
import numpy as np
```

```python
# Step 1 数据准备
x = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
            [1],
            [1],
            [0]])
```


```python
#Step 2 模型构建

num_epochs = 60000

#initialize weights
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1
```

```python
# 非线性函数（激活函数）
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))
```

```python
#Step 3 模型训练

for j in range(num_epochs):
    #feed forward through layers 0,1, and 2
    k0 = x  
    k1 = nonlin(np.dot(k0, syn0)) 
    k2 = nonlin(np.dot(k1, syn1))
    
    #how much did we miss the target value?
    k2_error = y - k2            
    
    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(k2_error))))
    
    #in what direction is the target value?
    k2_delta = k2_error*nonlin(k2, deriv=True) #
    
    #how much did each k1 value contribute to k2 error
    k1_error = k2_delta.dot(syn1.T)
    
    k1_delta= k1_error * nonlin(k1,deriv=True)
    
    syn1 += k1.T.dot(k2_delta)
    syn0 += k0.T.dot(k1_delta)
```

## Challenge 地震级数预测
The challenge for this video is to build a neural network to predict the magnitude(级数) of an Earthquake given the date, time, Latitude, and Longitude as features. [This](https://www.kaggle.com/usgs/earthquake-database) is the dataset. Optimize at least 1 hyperparameter using Random Search. See [this](http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html) example for more information.

You can use any library you like, *bonus points* are given if you do this using only numpy.

```python
# Step 1 Collect Data
N=1000
dataframe = pd.read_csv('database.csv')
x = dataframe[['Latitude','Longitude','Depth']].as_matrix()[:N]
y = dataframe['Magnitude'].as_matrix()[:N].reshape(N,1)

#Step 2 build model

num_epochs = 60000

#initialize weights
syn0 = 2*np.random.random((3,N)) - 1
syn1 = 2*np.random.random((N,1)) - 1

#Step 3 Train Model

for j in range(num_epochs):
    #feed forward through layers 0,1, and 2
    k0 = x  #输入层
    k1 = nonlin(np.dot(k0, syn0)) 
    k2 = nonlin(np.dot(k1, syn1)) 
    
    #how much did we miss the target value?
    k2_error = y - k2             
    
    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(k2_error))))
    
    #in what direction is the target value?
    k2_delta = k2_error*nonlin(k2, deriv=True) #
    
    #how much did each k1 value contribute to k2 error
    k1_error = k2_delta.dot(syn1.T)
    
    k1_delta= k1_error * nonlin(k1,deriv=True)
    
    syn1 += k1.T.dot(k2_delta)
    syn0 += k0.T.dot(k1_delta)
```
直接把23412维的输入放进这个两层的神经网络里了，然后就就死机了，，，明天再说吧。

```
RuntimeWarning: overflow encountered in exp
```