---
layout: post
title: 如何在线性回归中运用梯度下降法
date: 2018-2-1 19:35:14 +0800
categories: AI
tags: 深度学习 Siraj 
img: http://wangweiguang.xyz/images/ML.jpg
---



课程视频见：[https://www.bilibili.com/video/av19005521/](https://www.bilibili.com/video/av19005521/)

这周的直播部分介绍了梯度下降方法，和大多数深度学习入门教程都差不多，作巩固。

梯度下降法相关相关知识再这里就不写了，见[吴恩达深度学习课程第二周](http://wangweiguang.xyz/ai/2017/10/16/dl2.html#学习的方法)。

下面是实现代码：

```python
#本程序用来演示梯度下降的实现过程，梯度下降的作用是通过计算损失函数，最终得到模型中最优的m和b值

from numpy import *

# y = mx + b
def compute_error_for_line_given_points(b, m, points): #损失函数定义
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate): #每个梯度参数的更新
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations): #梯度下降函数
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run(): #“主函数”，导入数据，初始化参数，输出梯度下降后参数结果
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    run()

```

