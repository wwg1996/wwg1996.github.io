---
layout: post
title: Tensorflow的基本使用
date: 2018-2-4 19:15:22 +0800
categories: AI
tags: 深度学习 Siraj 
img: http://wangweiguang.xyz/images/ML.jpg
---

课程视频地址：[https://www.bilibili.com/video/av19129164/](https://www.bilibili.com/video/av19129164/)

# 预备

这是示例将说明如何在TensorFlow中对简单数据集进行分类。这里，我们正在建立一个模型来帮助朋友选择买房。她给了对下面这个表格里房子的评价。我们要建立一个模型，把房子的面积和浴室的数量作为输入，并输出一个关于她是否喜欢房子的预测。

| Area (sq ft) (x1) | Bathrooms (x2) | Label (y) |
| ----------------- | -------------- | --------- |
| 2,104             | 3              | Good      |
| 1,600             | 3              | Good      |
| 2,400             | 3              | Good      |
| 1,416             | 2              | Bad       |
| 3,000             | 4              | Bad       |
| 1,985             | 4              | Good      |
| 1,534             | 3              | Bad       |
| 1,427             | 3              | Good      |
| 1,380             | 3              | Good      |
| 1,494             | 3              | Good      |

 首先导入库

```python
%matplotlib inline               
import pandas as pd              # 让我们把数据作为表格形式来处理
import numpy as np               
import matplotlib.pyplot as plt  
import tensorflow as tf          # Fire from the gods 
```

然后我们将加载房屋数据CSV文件。pandas是一个出色的库，它为我们在处理类似表格的数据时提供了很大的灵活性。我们将表格(或csv文件，或excel表)加载到一个“dataframe”中，并按照我们喜欢的方式处理它。你可以把它看作是一种编程方式来做很多以前用Excel做的事情。

```python
dataframe = pd.read_csv("data.csv") 
dataframe = dataframe.drop(["index", "price", "sq_price"], axis=1) # 取出不需要的数据
dataframe = dataframe[0:10] # 这里只取十组数据作为样本集
dataframe
```
现在dataframe只有一些特征值，接下来给每组数据加上标签

```python
dataframe.loc[:, ("y1")] = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1] # This is our friend's list of which houses she liked
                                                          # 1 = good, 0 = bad
dataframe.loc[:, ("y2")] = dataframe["y1"] == 0           # y2为y1的相反数
dataframe.loc[:, ("y2")] = dataframe["y2"].astype(int)    # 把布尔变量转为int型变量
# y1表示喜欢这个房子，y2表示不喜欢
# (是的，这里y2是多余的。但是开始这样做会让你之后平滑过渡到多级分类)
dataframe 
```
所有我们需要的数据在dataframe里都准备好了，接下来我们需要将它变形为矩阵，用以输入TensorFlow

```python
inputX = dataframe.loc[:, ['area', 'bathrooms']].as_matrix()
inputY = dataframe.loc[:, ["y1", "y2"]].as_matrix()
```

定义一些训练过程用到的参数

```python
# Parameters
learning_rate = 0.000001
training_epochs = 2000
display_step = 50
n_samples = inputY.size

```
# 重点
And now to define the TensorFlow operations. Notice that this is a declaration step where we tell TensorFlow how the prediction is calculated. If we execute it, no calculation would be made. It would just acknowledge that it now knows how to do the operation.
现在定义TensorFlow的操作过程。这里是一个声明的步骤，这里将告诉TensorFlow预测过程要如何进行计算。执行下面的语句，并不会进行任何运算。我们只是告诉了他如何进行操作。（运算会在具体的数据输入后进行）

```python
x = tf.placeholder(tf.float32, [None, 2])   # 告诉TensorFlow我们接下来将会输入一组样本集。
                                            # 样本集的形状是[None, 2]，意思是每组（行）样本包含两个数值（面积和浴室数）
                                            # 列是None，因为不确定有多少组样本，因此样本的个数没有限制。
            
W = tf.Variable(tf.zeros([2, 2]))           # 初始化权重集为2×2的零矩阵
                                            # 之后的训练过程会不断对其进行更新
    
b = tf.Variable(tf.zeros([2]))              # 初始化偏移量

y_values = tf.add(tf.matmul(x, W), b)       # 定义运算方程式 x*W + b （这里xW是矩阵乘法）
    
y = tf.nn.softmax(y_values)                 # 这里用softmax作为“激活函数”，会将前一层输出的数字转换为概率形式
    
y_ = tf.placeholder(tf.float32, [None,2])   # 因为要训练，还需要输入标签矩阵

```

下面明确损失函数并且使用梯度下降法

```python

# 损失函数: 平方和误差
cost = tf.reduce_sum(tf.pow(y_ - y, 2))/(2*n_samples)
# 优化方法：梯度下降法
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
```

```python
# 初始化变量和tensorflow工作段
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
```

预备部分完结撒花

现在可以开始训练了

```python
for i in range(training_epochs):  
    sess.run(optimizer, feed_dict={x: inputX, y_: inputY}) 
    # 输入样本集，包括标签集，梯度下降法作为优化器，run起来
    
    # 到这里机器学习部分就全部完成了! 下面可以输出一下调试信息 
    # 每一阶段的显示日志
    if (i) % display_step == 0:
        cc = sess.run(cost, feed_dict={x: inputX, y_:inputY})
        print("Training step:", '%04d' % (i), "cost=", "{:.9f}".format(cc),"W=", sess.run(W), "b=", sess.run(b))

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={x: inputX, y_: inputY})
print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

```

现在训练结束了。TensorFlow此时保留了我们的训练模型(基本上就是定义的操作，加上训练过程中产生的变量W和b)。
可以得到训练后的损失值是0.109537，比第一次迭代的成本值(0.114958666)好。让我们在数据集上应用这个模型，看看它是如何工作的:

```python
sess.run(y, feed_dict={x: inputX })
```

```
output：
array([[ 0.71125221,  0.28874779],
       [ 0.66498977,  0.33501023],
       [ 0.73657656,  0.26342347],
       [ 0.64718789,  0.35281211],
       [ 0.78335613,  0.2166439 ],
       [ 0.70069474,  0.29930523],
       [ 0.65866327,  0.34133676],
       [ 0.64828628,  0.35171372],
       [ 0.64368278,  0.35631716],
       [ 0.65480113,  0.3451989 ]], dtype=float32)
```

上面的结果显示了模型对十组样本的预测，每组预测包括两个值，So It's “好”的概率，第二个是“坏”的概率。这里预测的都是好的房子（好的概率大），就是0.7的正确率了，当然因为样本集和模型都很简单，加一些隐含层并且用上所有的样本会更好，不过已经解释了使用TensorFlow的基本流程。

