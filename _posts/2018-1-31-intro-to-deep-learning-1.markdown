---
layout: post
title: How to Make a Prediction
date:  17:33:07 +0800
categories: AI
tags: Siraj  深度学习 
img: http://wangweiguang.xyz/images/ML.jpg
---

课程视频见[https://www.bilibili.com/video/av18910318/](https://www.bilibili.com/video/av18910318/)



* 
{:toc}
# 1 Intro to Deep Learning

本周总结：

* 机器学习——定义期望的输出，然后让我们的算法自己学习到到达那里的步骤

* 3种学习方式——监督学习，无监督学习，强化学习

* 线性回归模型显示了变量间的独立性关系，创造最佳拟合的线

  第一周基础没什么好说的，编个小程序热热身。

## 实例

```python
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data
dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize reasults
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
```

## Challenge

The challenge for this video is to use scikit-learn to create a line of best fit for the included 'challenge_dataset'. Then, make a prediction for an existing data point and see how close it matches up to the actual value. Print out the error you get. You can use scikit-learn's documentation for more help. 

Bonus points if you perform linear regression on a dataset with 3 different variables

```python
# Mine

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

# read data
df = pd.read_csv('challenge_dataset.txt', names=['X','Y'])
x_values = df[['X']]
y_values = df[['Y']]

# train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

# visualize reasults
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()

# average error
np_data = np.array(df['Y'])
np.mean(np.abs(np_data - body_reg.predict(x_values)))
```

## Ludo's winning code:

```python
%matplotlib inline

# Imports
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import sklearn
import numpy as np
```

### Regression with 2 variables

#### 1. Import and visualize data

```python
df = pd.read_csv('challenge_dataset.txt', names=['X','Y'])
```

#### 2. 2D Regression 

```python
from sklearn.model_selection import train_test_split

# 随机划分训练集和测试集
X_train, X_test, y_train, y_test = np.asarray(train_test_split(df['X'], df['Y'], test_size=0.1))

from sklearn.linear_model import LinearRegression

# 创建线性回归模型，训练模型
reg = LinearRegression()
reg.fit(X_train.values.reshape(-1,1), y_train.values.reshape(-1,1))

# 测试集测试
print('Score: ', reg.score(X_test.values.reshape(-1,1), y_test.values.reshape(-1,1)))
```

#### 3. Plot regression and visualize results

```python
x_line = np.arange(5,25).reshape(-1,1)
sns.regplot(x=df['X'], y=df['Y'], data=df, fit_reg=False)
plt.plot(x_line, reg.predict(x_line))
plt.show()
```

详细代码见：[https://github.com/ludobouan/linear-regression-sklearn](https://github.com/ludobouan/linear-regression-sklearn)

### 小结

这系列深度学习课程真是挺不错的，每天听一节，自己做一下挑战题，再看看每周榜首的代码。一边学习常见的几个库，一边再巩固Python基础语法，一个假期应该会提示不少的。